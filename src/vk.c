#include "vk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vulkan/vulkan.h>

#include "gguf.h"
#include "shaders.h"

#define ACAP (320u * 1024u)      /* activation words per sequence */
#define XQ_RES ACAP              /* reserved: q8k of the MoE input */
#define ABUF (ACAP + 2048u)
#define Q8W ((uint32_t)(sizeof(q8k) / 4))
#define BMAX 64
#define NONE 0xffffffffu
#define PIPE_NORM 20
#define PIPE_CONV 21
#define PIPE_SCAN 22
#define UPCHUNK (32u << 20)

static double
vnow(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}
double vk_t_wait, vk_t_sub;
long vk_n_sub;

typedef struct {
	uint32_t w_off, row_words, K, N, x_off, y_off;
} pushc;

static struct {
	int ok;
	VkInstance inst;
	VkPhysicalDevice pd;
	VkDevice dev;
	VkQueue q;
	uint32_t qf;
	char name[256];
	VkPhysicalDeviceMemoryProperties mp;

	VkBuffer wbuf, abuf, hbuf, ubuf, sbuf;
	VkDeviceMemory wmem, amem, hmem, umem, smem;
	size_t scap, sused;           /* state buffer, floats */
	size_t wcap, wused;
	float *hmap;
	uint8_t *umap;

	VkDescriptorSetLayout dsl;
	VkPipelineLayout pl;
	VkDescriptorPool dp;
	VkDescriptorSet ds;
	VkPipeline pipe[32];
	VkCommandPool cp;
	VkCommandBuffer cb;
	VkFence fence;
	uint32_t rows_per_wg, sg_per_wg;
	int dot;                /* native packed int8 dot product */

	/* plain matvec batch */
	int nx, nd;
	struct { const wt *w; int e; const float *x; const q8k *xq; float *y; } d[BMAX];
	/* recording */
	uint32_t aused, cur;
	int nup, upw, ndl;
	VkBufferCopy up[BMAX];
	struct { uint32_t off, n; float *dst; } dl[BMAX];
	int inflight;
	/* fused paths */
	struct lay { int ok; uint32_t an, fn, cw, cb, dtb, a, d, sn, sconv, sssm; } *lay;
	int nlay;
	uint32_t xq_moe, x_out, xn_out;
	int nmoe;
	struct { const wt *g, *u, *d; int e; float *out; } moe[16];
	/* Q2K_GPUPROF: per-stage GPU timestamps of the fused mixer */
	VkQueryPool qp;
	int prof, nts;
	float ts_period;
	double stage[16];
	long nstage;
} V;

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
	fprintf(stderr, "vulkan: %s failed (%d)\n", #x, (int)r_); return -1; } } while (0)

int
vk_ready(void)
{
	return V.ok;
}

const char *
vk_device_name(void)
{
	return V.name;
}

size_t
vk_used(void)
{
	return V.wused;
}

static int
memtype(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid)
{
	uint32_t i;
	for (i = 0; i < V.mp.memoryTypeCount; i++)
		if ((bits & (1u << i)) &&
		    (V.mp.memoryTypes[i].propertyFlags & want) == want &&
		    !(V.mp.memoryTypes[i].propertyFlags & avoid))
			return (int)i;
	return -1;
}

static int
mkbuf(VkDeviceSize size, VkBufferUsageFlags use, int host, VkBuffer *b,
    VkDeviceMemory *m, void **map)
{
	VkBufferCreateInfo bi;
	VkMemoryRequirements mr;
	VkMemoryAllocateInfo ai;
	int t = -1;

	memset(&bi, 0, sizeof(bi));
	bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bi.size = size;
	bi.usage = use;
	bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	CK(vkCreateBuffer(V.dev, &bi, NULL, b));
	vkGetBufferMemoryRequirements(V.dev, *b, &mr);
	if (host) {
		/* cached host memory reads back fast; fall back to plain */
		t = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
		if (t < 0)
			t = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
	} else {
		t = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
		if (t < 0)
			t = memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
	}
	if (t < 0) {
		fprintf(stderr, "vulkan: no suitable memory type\n");
		return -1;
	}
	memset(&ai, 0, sizeof(ai));
	ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ai.allocationSize = mr.size;
	ai.memoryTypeIndex = (uint32_t)t;
	CK(vkAllocateMemory(V.dev, &ai, NULL, m));
	CK(vkBindBufferMemory(V.dev, *b, *m, 0));
	if (map)
		CK(vkMapMemory(V.dev, *m, 0, VK_WHOLE_SIZE, 0, map));
	return 0;
}

static int
mkpipe(uint32_t type, const uint32_t *code, size_t bytes)
{
	VkShaderModuleCreateInfo si;
	VkComputePipelineCreateInfo ci;
	VkShaderModule sm;
	VkResult r;

	memset(&si, 0, sizeof(si));
	si.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	si.codeSize = bytes;
	si.pCode = code;
	CK(vkCreateShaderModule(V.dev, &si, NULL, &sm));
	memset(&ci, 0, sizeof(ci));
	ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	ci.stage.module = sm;
	ci.stage.pName = "main";
	ci.layout = V.pl;
	r = vkCreateComputePipelines(V.dev, VK_NULL_HANDLE, 1, &ci, NULL, &V.pipe[type]);
	vkDestroyShaderModule(V.dev, sm, NULL);
	CK(r);
	return 0;
}

int
vk_init(size_t budget, size_t state_bytes, int devidx, int verbose)
{
	VkApplicationInfo app;
	VkInstanceCreateInfo ici;
	VkPhysicalDevice pds[16];
	uint32_t npd = 16, i, nqf = 0, pick = 0;
	int best = -1;
	VkPhysicalDeviceProperties pp;
	VkPhysicalDeviceSubgroupProperties sp;
	VkPhysicalDeviceProperties2 pp2;
	VkQueueFamilyProperties qfp[16];
	float prio = 1;
	VkDeviceQueueCreateInfo qi;
	VkDeviceCreateInfo dci;
	VkDescriptorSetLayoutBinding lb[4];
	VkDescriptorSetLayoutCreateInfo lci;
	VkPushConstantRange pr;
	VkPipelineLayoutCreateInfo plci;
	VkDescriptorPoolSize ps;
	VkDescriptorPoolCreateInfo dpci;
	VkDescriptorSetAllocateInfo dsai;
	VkDescriptorBufferInfo dbi[4];
	VkWriteDescriptorSet wr[4];
	VkCommandPoolCreateInfo cpci;
	VkCommandBufferAllocateInfo cbai;
	VkFenceCreateInfo fci;
	VkPhysicalDeviceShaderIntegerDotProductFeatures idf;
	const char *exts[2];
	uint32_t next = 0;
	int has_budget = 0;
	void *p;

	memset(&V, 0, sizeof(V));
	memset(&app, 0, sizeof(app));
	app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app.pApplicationName = "q2k";
	app.apiVersion = VK_API_VERSION_1_1;
	memset(&ici, 0, sizeof(ici));
	ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	ici.pApplicationInfo = &app;
	CK(vkCreateInstance(&ici, NULL, &V.inst));
	CK(vkEnumeratePhysicalDevices(V.inst, &npd, pds));
	if (npd == 0) {
		fprintf(stderr, "vulkan: no devices\n");
		return -1;
	}
	if (devidx >= 0 && (uint32_t)devidx < npd)
		pick = (uint32_t)devidx;
	else {
		/* discrete > integrated > virtual > anything but CPU */
		for (i = 0; i < npd; i++) {
			int score;
			vkGetPhysicalDeviceProperties(pds[i], &pp);
			score = pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 4 :
			    pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 3 :
			    pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU ? 2 :
			    pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? 0 : 1;
			if (score > best) {
				best = score;
				pick = i;
			}
		}
	}
	V.pd = pds[pick];
	vkGetPhysicalDeviceProperties(V.pd, &pp);
	snprintf(V.name, sizeof(V.name), "%s", pp.deviceName);
	vkGetPhysicalDeviceMemoryProperties(V.pd, &V.mp);

	memset(&sp, 0, sizeof(sp));
	sp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
	memset(&pp2, 0, sizeof(pp2));
	pp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	pp2.pNext = &sp;
	vkGetPhysicalDeviceProperties2(V.pd, &pp2);
	if (!(sp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) ||
	    !(sp.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ||
	    sp.subgroupSize == 0 || 256 % sp.subgroupSize) {
		fprintf(stderr, "vulkan: %s lacks subgroup arithmetic\n", V.name);
		return -1;
	}
	V.rows_per_wg = 256 / sp.subgroupSize;
	V.sg_per_wg = 256 / sp.subgroupSize;

	vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &nqf, NULL);
	if (nqf > 16)
		nqf = 16;
	vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &nqf, qfp);
	V.qf = UINT32_MAX;
	for (i = 0; i < nqf; i++)
		if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			/* prefer a compute queue without graphics */
			if (V.qf == UINT32_MAX || !(qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
				V.qf = i;
		}
	if (V.qf == UINT32_MAX) {
		fprintf(stderr, "vulkan: no compute queue\n");
		return -1;
	}
	memset(&qi, 0, sizeof(qi));
	qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	qi.queueFamilyIndex = V.qf;
	qi.queueCount = 1;
	qi.pQueuePriorities = &prio;
	/* packed int8 dot product (DP4A) when the device has it */
	{
		uint32_t ne = 0;
		VkExtensionProperties *ep;
		vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, NULL);
		ep = calloc(ne ? ne : 1, sizeof(*ep));
		if (ep && vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, ep) == VK_SUCCESS)
			for (i = 0; i < ne; i++) {
				if (!strcmp(ep[i].extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
					V.dot = 1;
				else if (!strcmp(ep[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
					has_budget = 1;
			}
		free(ep);
	}
	memset(&idf, 0, sizeof(idf));
	idf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
	if (V.dot) {
		VkPhysicalDeviceFeatures2 f2;
		memset(&f2, 0, sizeof(f2));
		f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		f2.pNext = &idf;
		vkGetPhysicalDeviceFeatures2(V.pd, &f2);
		V.dot = idf.shaderIntegerDotProduct == VK_TRUE;
	}
	if (getenv("Q2K_NO_DOT"))
		V.dot = 0;
	memset(&dci, 0, sizeof(dci));
	dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qi;
	if (V.dot) {
		dci.pNext = &idf;
		exts[next++] = VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
	}
	if (has_budget)
		exts[next++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
	dci.enabledExtensionCount = next;
	dci.ppEnabledExtensionNames = exts;
	CK(vkCreateDevice(V.pd, &dci, NULL, &V.dev));
	vkGetDeviceQueue(V.dev, V.qf, 0, &V.q);

	/* keep every allocation of ours inside the VRAM that is free:
	 * at most VK_VRAM_LIMIT, and the driver's budget for the heap when
	 * VK_EXT_memory_budget reports it */
	{
		size_t limit = VK_VRAM_LIMIT, other = (size_t)ABUF * 4 + state_bytes + ((size_t)32 << 20);
		if (has_budget) {
			VkPhysicalDeviceMemoryBudgetPropertiesEXT mb;
			VkPhysicalDeviceMemoryProperties2 mp2;
			uint32_t h, t;
			memset(&mb, 0, sizeof(mb));
			mb.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
			memset(&mp2, 0, sizeof(mp2));
			mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
			mp2.pNext = &mb;
			vkGetPhysicalDeviceMemoryProperties2(V.pd, &mp2);
			/* the heap our device-local buffers come from */
			for (t = 0; t < V.mp.memoryTypeCount; t++)
				if ((V.mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
				    !(V.mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
					break;
			h = t < V.mp.memoryTypeCount ? V.mp.memoryTypes[t].heapIndex : 0;
			if (mb.heapBudget[h] && mb.heapBudget[h] - mb.heapUsage[h] < limit)
				limit = (size_t)(mb.heapBudget[h] - mb.heapUsage[h]);
		}
		V.wcap = budget;
		if (V.wcap + other > limit) {
			V.wcap = limit > other ? limit - other : 0;
			fprintf(stderr, "vulkan: weight budget clamped to %zu MiB (VRAM available %zu MiB)\n",
			    V.wcap >> 20, limit >> 20);
		}
		if (V.wcap < 4096)
			V.wcap = 4096;
	}
	if (mkbuf(V.wcap, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	    0, &V.wbuf, &V.wmem, NULL))
		return -1;
	if (mkbuf((VkDeviceSize)ABUF * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	    0, &V.abuf, &V.amem, NULL))
		return -1;
	if (mkbuf((VkDeviceSize)ABUF * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 1, &V.hbuf, &V.hmem, &p))
		return -1;
	V.scap = state_bytes / 4;
	if (mkbuf(state_bytes > 256 ? state_bytes : 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	    VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, &V.sbuf, &V.smem, NULL))
		return -1;
	V.hmap = p;

	memset(lb, 0, sizeof(lb));
	for (i = 0; i < 4; i++) {
		lb[i].binding = i;
		lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		lb[i].descriptorCount = 1;
		lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	memset(&lci, 0, sizeof(lci));
	lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	lci.bindingCount = 4;
	lci.pBindings = lb;
	CK(vkCreateDescriptorSetLayout(V.dev, &lci, NULL, &V.dsl));
	pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pr.offset = 0;
	pr.size = 64;
	memset(&plci, 0, sizeof(plci));
	plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plci.setLayoutCount = 1;
	plci.pSetLayouts = &V.dsl;
	plci.pushConstantRangeCount = 1;
	plci.pPushConstantRanges = &pr;
	CK(vkCreatePipelineLayout(V.dev, &plci, NULL, &V.pl));
	ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	ps.descriptorCount = 4;
	memset(&dpci, 0, sizeof(dpci));
	dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpci.maxSets = 1;
	dpci.poolSizeCount = 1;
	dpci.pPoolSizes = &ps;
	CK(vkCreateDescriptorPool(V.dev, &dpci, NULL, &V.dp));
	memset(&dsai, 0, sizeof(dsai));
	dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsai.descriptorPool = V.dp;
	dsai.descriptorSetCount = 1;
	dsai.pSetLayouts = &V.dsl;
	CK(vkAllocateDescriptorSets(V.dev, &dsai, &V.ds));
	dbi[0].buffer = V.wbuf;
	dbi[1].buffer = dbi[2].buffer = V.abuf;
	dbi[3].buffer = V.sbuf;
	for (i = 0; i < 4; i++) {
		dbi[i].offset = 0;
		dbi[i].range = VK_WHOLE_SIZE;
		memset(&wr[i], 0, sizeof(wr[i]));
		wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		wr[i].dstSet = V.ds;
		wr[i].dstBinding = i;
		wr[i].descriptorCount = 1;
		wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		wr[i].pBufferInfo = &dbi[i];
	}
	vkUpdateDescriptorSets(V.dev, 4, wr, 0, NULL);

#define PIPE(t, a) (V.dot ? mkpipe(t, a, sizeof(a)) : mkpipe(t, a##_nd, sizeof(a##_nd)))
	if (PIPE(GT_Q2_K, spv_q2_k) || PIPE(GT_Q3_K, spv_q3_k) ||
	    PIPE(GT_Q4_K, spv_q4_k) || PIPE(GT_Q5_K, spv_q5_k) ||
	    PIPE(GT_Q6_K, spv_q6_k) || PIPE(GT_Q8_0, spv_q8_0))
		return -1;
#undef PIPE
	if (mkpipe(PIPE_NORM, spv_op_norm, sizeof(spv_op_norm)) ||
	    mkpipe(PIPE_CONV, spv_op_conv, sizeof(spv_op_conv)) ||
	    mkpipe(PIPE_SCAN, spv_op_scan, sizeof(spv_op_scan)))
		return -1;

	memset(&cpci, 0, sizeof(cpci));
	cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	cpci.queueFamilyIndex = V.qf;
	CK(vkCreateCommandPool(V.dev, &cpci, NULL, &V.cp));
	memset(&cbai, 0, sizeof(cbai));
	cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cbai.commandPool = V.cp;
	cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbai.commandBufferCount = 1;
	CK(vkAllocateCommandBuffers(V.dev, &cbai, &V.cb));
	memset(&fci, 0, sizeof(fci));
	fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	CK(vkCreateFence(V.dev, &fci, NULL, &V.fence));

	if (getenv("Q2K_GPUPROF") && pp.limits.timestampComputeAndGraphics) {
		VkQueryPoolCreateInfo qi2;
		memset(&qi2, 0, sizeof(qi2));
		qi2.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		qi2.queryType = VK_QUERY_TYPE_TIMESTAMP;
		qi2.queryCount = 16;
		if (vkCreateQueryPool(V.dev, &qi2, NULL, &V.qp) == VK_SUCCESS) {
			V.prof = 1;
			V.ts_period = pp.limits.timestampPeriod;
		}
	}
	if (verbose)
		fprintf(stderr, "vulkan: %s, subgroup %u, %s dot, arena %zu MiB\n",
		    V.name, sp.subgroupSize, V.dot ? "dp4a" : "emulated", V.wcap >> 20);
	V.ok = 1;
	return 0;
}

void
vk_shutdown(void)
{
	int i;
	if (!V.dev)
		return;
	vkDeviceWaitIdle(V.dev);
	for (i = 0; i < 32; i++)
		if (V.pipe[i])
			vkDestroyPipeline(V.dev, V.pipe[i], NULL);
	if (V.fence) vkDestroyFence(V.dev, V.fence, NULL);
	if (V.cp) vkDestroyCommandPool(V.dev, V.cp, NULL);
	if (V.dp) vkDestroyDescriptorPool(V.dev, V.dp, NULL);
	if (V.pl) vkDestroyPipelineLayout(V.dev, V.pl, NULL);
	if (V.dsl) vkDestroyDescriptorSetLayout(V.dev, V.dsl, NULL);
	if (V.wbuf) vkDestroyBuffer(V.dev, V.wbuf, NULL);
	if (V.abuf) vkDestroyBuffer(V.dev, V.abuf, NULL);
	if (V.hbuf) vkDestroyBuffer(V.dev, V.hbuf, NULL);
	if (V.ubuf) vkDestroyBuffer(V.dev, V.ubuf, NULL);
	if (V.sbuf) vkDestroyBuffer(V.dev, V.sbuf, NULL);
	if (V.smem) vkFreeMemory(V.dev, V.smem, NULL);
	free(V.lay);
	if (V.wmem) vkFreeMemory(V.dev, V.wmem, NULL);
	if (V.amem) vkFreeMemory(V.dev, V.amem, NULL);
	if (V.hmem) vkFreeMemory(V.dev, V.hmem, NULL);
	if (V.umem) vkFreeMemory(V.dev, V.umem, NULL);
	vkDestroyDevice(V.dev, NULL);
	vkDestroyInstance(V.inst, NULL);
	memset(&V, 0, sizeof(V));
}

/* ---------------------------------------------------------------- */
/* uploads                                                          */

/* GPU block size: blocks padded to a multiple of 4 bytes */
static size_t
gpu_block_bytes(uint32_t t)
{
	return (ggml_type_bytes(t) + 3) & ~(size_t)3;
}

/* repack n blocks of type t from src to dst (GPU layout) */
static void
repack(uint32_t t, const uint8_t *src, uint8_t *dst, size_t n)
{
	size_t sb = ggml_type_bytes(t), db = gpu_block_bytes(t), i;
	if (sb == db) {
		memcpy(dst, src, n * sb);
		return;
	}
	for (i = 0; i < n; i++, src += sb, dst += db) {
		if (t == GT_Q8_0) {
			memcpy(dst, src + 2, 32);    /* qs first */
			memcpy(dst + 32, src, 2);    /* then d */
			dst[34] = dst[35] = 0;
		} else {
			memcpy(dst, src, sb);
			memset(dst + sb, 0, db - sb);
		}
	}
}

static int
submit_wait(void)
{
	VkSubmitInfo si;
	memset(&si, 0, sizeof(si));
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &V.cb;
	CK(vkQueueSubmit(V.q, 1, &si, V.fence));
	CK(vkWaitForFences(V.dev, 1, &V.fence, VK_TRUE, UINT64_MAX));
	CK(vkResetFences(V.dev, 1, &V.fence));
	return 0;
}

static int
flush_upload(size_t n, uint64_t dst)
{
	VkCommandBufferBeginInfo bi;
	VkBufferCopy rg;

	if (n == 0)
		return 0;
	memset(&bi, 0, sizeof(bi));
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	CK(vkBeginCommandBuffer(V.cb, &bi));
	rg.srcOffset = 0;
	rg.dstOffset = dst;
	rg.size = n;
	vkCmdCopyBuffer(V.cb, V.ubuf, V.wbuf, 1, &rg);
	CK(vkEndCommandBuffer(V.cb));
	return submit_wait();
}

static size_t
expert_bytes(const wt *w)
{
	size_t sb = ggml_type_bytes(w->type), db = gpu_block_bytes(w->type);
	return (size_t)w->N * (w->row_bytes / sb * db);
}

int
vk_upload(wt *w, const int *ex, int n)
{
	size_t sb, db, rb, grb, eb, fill = 0, total;
	uint64_t base, flushed;
	int k, r;
	void *p;

	if (!V.ok || !V.pipe[w->type] || n <= 0 || w->gpu_n)
		return -1;
	sb = ggml_type_bytes(w->type);
	db = gpu_block_bytes(w->type);
	rb = w->row_bytes;
	grb = rb / sb * db;
	eb = expert_bytes(w);
	total = eb * (size_t)n;
	if (V.wused + total > V.wcap)
		return -1;
	if (!w->gslot && !(w->gslot = malloc(sizeof(int) * (size_t)w->E)))
		return -1;
	if (!V.ubuf) {
		if (mkbuf(UPCHUNK, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 1, &V.ubuf, &V.umem, &p))
			return -1;
		V.umap = p;
	}
	base = flushed = V.wused;
	for (k = 0; k < w->E; k++)
		w->gslot[k] = -1;
	/* experts packed in slot order */
	for (k = 0; k < n; k++) {
		const uint8_t *src = w->data + (size_t)ex[k] * (size_t)w->N * rb;
		for (r = 0; r < w->N; r++) {
			if (fill + grb > UPCHUNK) {
				if (flush_upload(fill, flushed))
					return -1;
				flushed += fill;
				fill = 0;
			}
			repack(w->type, src + (size_t)r * rb, V.umap + fill, rb / sb);
			fill += grb;
		}
		w->gslot[ex[k]] = k;
	}
	if (flush_upload(fill, flushed))
		return -1;
	w->gpu_n = n;
	w->gpu_off = base;
	w->gpu_row_bytes = grb;
	V.wused += (total + 255) & ~(size_t)255;
	return 0;
}

/* a small f32 vector into the arena; returns its word offset or NONE */
static uint32_t
upload_raw(const void *src, size_t bytes)
{
	uint64_t off = V.wused;
	void *p;
	if (V.wused + bytes > V.wcap)
		return NONE;
	if (!V.ubuf) {
		if (mkbuf(UPCHUNK, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 1, &V.ubuf, &V.umem, &p))
			return NONE;
		V.umap = p;
	}
	memcpy(V.umap, src, bytes);
	if (flush_upload((bytes + 3) & ~(size_t)3, off))
		return NONE;
	V.wused += (bytes + 255) & ~(size_t)255;
	return (uint32_t)(off / 4);
}

/* norms, conv and SSM parameters of a Mamba2 layer whose projections
 * are resident, and its state slots: enables vk_mamba_layer */
static void
setup_mamba(const model *m, int il)
{
	const hparams *h = &m->hp;
	const layer *L = &m->L[il];
	struct lay *ly = &V.lay[il];
	size_t E = (size_t)h->n_embd, CH = (size_t)h->conv_ch, DI = (size_t)h->d_inner;
	size_t need = 3 * CH + DI * (size_t)h->d_state;

	if (L->attn || !L->ssm_in.gpu_n || !L->ssm_out.gpu_n || V.sused + need > V.scap)
		return;
	ly->an = upload_raw(L->attn_norm, E * 4);
	ly->fn = upload_raw(L->ffn_norm, E * 4);
	ly->cw = upload_raw(L->conv_w, 4 * CH * 4);
	ly->cb = upload_raw(L->conv_b, CH * 4);
	ly->dtb = upload_raw(L->dt_b, (size_t)h->ssm_heads * 4);
	ly->a = upload_raw(L->A, (size_t)h->ssm_heads * 4);
	ly->d = upload_raw(L->D, (size_t)h->ssm_heads * 4);
	ly->sn = upload_raw(L->ssm_norm, DI * 4);
	if (ly->an == NONE || ly->fn == NONE || ly->cw == NONE || ly->cb == NONE ||
	    ly->dtb == NONE || ly->a == NONE || ly->d == NONE || ly->sn == NONE)
		return;
	ly->sconv = (uint32_t)V.sused;
	ly->sssm = (uint32_t)(V.sused + 3 * CH);
	V.sused += need;
	ly->ok = 1;
}

static int
by_size(const void *a, const void *b)
{
	const wt *x = *(wt *const *)a, *y = *(wt *const *)b;
	return x->nbytes < y->nbytes ? 1 : x->nbytes > y->nbytes ? -1 : 0;
}

/*
 * Placement is by whole tensor and whole expert, fixed at load:
 *  1. dense tensors (read every token: mixers, attention, shared
 *     experts, the tied output), largest first;
 *  2. then routed experts, each as its gate+up+down triple (they are
 *     used together), the same number per layer, in rounds over the
 *     layers, until the budget is spent.
 * Every matvec then runs entirely where its weights live; a batch's
 * GPU part runs alongside its CPU part.
 */
void
vk_plan_and_upload(model *m, int verbose)
{
	wt *dense[512];
	int nd = 0, i, l, nl = m->hp.n_layer, ne = m->hp.n_expert;
	int **sel = NULL, *cnt = NULL, round, full = 0;
	double hot = 0, hot_gpu = 0;
	size_t dense_b = 0, exp_b = 0;
	int nexp = 0, ndense = 0;

	for (i = 0; i < m->n_all; i++)
		if (m->all[i]->heat >= 1)
			dense[nd++] = m->all[i];
	qsort(dense, (size_t)nd, sizeof(wt *), by_size);
	for (i = 0; i < nd; i++) {
		int zero = 0;
		size_t before = V.wused;
		if (vk_upload(dense[i], &zero, 1) == 0) {
			ndense++;
			dense_b += V.wused - before;
		}
	}
	/* fused mixer for every Mamba2 layer whose projections fit */
	V.nlay = nl;
	V.lay = calloc((size_t)nl, sizeof(*V.lay));
	for (l = 0; V.lay && l < nl; l++)
		setup_mamba(m, l);
	if (V.lay)
		vk_state_reset();

	/* experts: choose per layer in rounds, then upload per tensor */
	sel = calloc((size_t)nl, sizeof(*sel));
	cnt = calloc((size_t)nl, sizeof(*cnt));
	if (!sel || !cnt)
		goto out;
	for (l = 0; l < nl; l++)
		if (!(sel[l] = malloc(sizeof(int) * (size_t)ne)))
			goto out;
	{
		size_t left = V.wcap > V.wused ? V.wcap - V.wused : 0;
		for (round = 0; round < ne && !full; round++)
			for (l = 0; l < nl && !full; l++) {
				const layer *L = &m->L[l];
				size_t need = expert_bytes(&L->gate_e) + expert_bytes(&L->up_e) +
				    expert_bytes(&L->down_e) + 3 * 256;
				if (need > left) {
					full = 1;
					break;
				}
				left -= need;
				sel[l][cnt[l]++] = round;
			}
	}
	for (l = 0; l < nl; l++) {
		layer *L = &m->L[l];
		size_t before = V.wused;
		if (!cnt[l])
			continue;
		vk_upload(&L->gate_e, sel[l], cnt[l]);
		vk_upload(&L->up_e, sel[l], cnt[l]);
		vk_upload(&L->down_e, sel[l], cnt[l]);
		exp_b += V.wused - before;
		nexp += cnt[l];
	}
out:
	if (sel)
		for (l = 0; l < nl; l++)
			free(sel[l]);
	free(sel);
	free(cnt);
	for (i = 0; i < m->n_all; i++) {
		const wt *w = m->all[i];
		hot += w->heat * (double)w->nbytes;
		hot_gpu += w->heat * (double)w->nbytes * w->gpu_n / w->E;
	}
	if (verbose)
		fprintf(stderr, "vulkan: resident: %d dense tensors (%.1f MiB) + %d experts "
		    "(%.1f/layer, %.1f MiB) = %.1f MiB; %.0f%% of per-token weight bytes "
		    "(%.0f of %.0f MiB)\n", ndense, dense_b / 1048576.0,
		    nexp, (double)nexp / nl, exp_b / 1048576.0, V.wused / 1048576.0,
		    100.0 * hot_gpu / hot, hot_gpu / 1048576.0, hot / 1048576.0);
}

/* ---------------------------------------------------------------- */
/* command sequences                                                */

/*
 * Everything the GPU does per token is a recorded sequence in one
 * command buffer: inputs are staged in the host-visible mirror of the
 * activation buffer and copied in, dispatches run with barriers
 * between dependent steps, outputs are copied back and handed to their
 * destinations by vk_wait.  Activation space is a bump allocator, reset
 * per sequence; host and device copies use the same offsets.
 */

static void
barrier(VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags ss, VkPipelineStageFlags ds)
{
	VkMemoryBarrier mb;
	memset(&mb, 0, sizeof(mb));
	mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
	mb.srcAccessMask = src;
	mb.dstAccessMask = dst;
	vkCmdPipelineBarrier(V.cb, ss, ds, 0, 1, &mb, 0, NULL, 0, NULL);
}

static void
ts(void)
{
	if (V.prof)
		vkCmdWriteTimestamp(V.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, V.qp, (uint32_t)V.nts++);
}

/* compute -> compute */
static void
cbar(void)
{
	ts();
	barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
	    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

static void
seq_begin(void)
{
	VkCommandBufferBeginInfo bi;
	memset(&bi, 0, sizeof(bi));
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(V.cb, &bi);
	vkCmdBindDescriptorSets(V.cb, VK_PIPELINE_BIND_POINT_COMPUTE, V.pl, 0, 1, &V.ds, 0, NULL);
	V.aused = 0;
	V.ndl = 0;
	V.cur = UINT32_MAX;
	V.nup = 0;
	V.upw = 0;
}

static uint32_t
seq_alloc(uint32_t words)
{
	uint32_t o = V.aused;
	V.aused += (words + 3) & ~3u;
	if (V.aused > ACAP) {
		fprintf(stderr, "vulkan: activation buffer overflow\n");
		exit(1);
	}
	return o;
}

/* stage words for upload; the copies are recorded by seq_flush_up */
static uint32_t
seq_put(const void *src, uint32_t words)
{
	uint32_t o = seq_alloc(words);
	memcpy(V.hmap + o, src, (size_t)words * 4);
	if (V.nup && V.up[V.nup - 1].srcOffset + V.up[V.nup - 1].size == (VkDeviceSize)o * 4)
		V.up[V.nup - 1].size += (VkDeviceSize)((words + 3) & ~3u) * 4;
	else {
		V.up[V.nup].srcOffset = V.up[V.nup].dstOffset = (VkDeviceSize)o * 4;
		V.up[V.nup].size = (VkDeviceSize)((words + 3) & ~3u) * 4;
		V.nup++;
	}
	return o;
}

static void
seq_flush_up(void)
{
	if (!V.nup)
		return;
	vkCmdCopyBuffer(V.cb, V.hbuf, V.abuf, (uint32_t)V.nup, V.up);
	barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
	    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	V.nup = 0;
}

static void
seq_pipe(uint32_t id)
{
	if (V.cur != id) {
		V.cur = id;
		vkCmdBindPipeline(V.cb, VK_PIPELINE_BIND_POINT_COMPUTE, V.pipe[id]);
	}
}

static void
seq_matvec(const wt *w, int e, uint32_t xq, uint32_t y)
{
	pushc pc;
	pc.w_off = (uint32_t)((w->gpu_off + (uint64_t)w->gslot[e] * (uint64_t)w->N *
	    w->gpu_row_bytes) / 4);
	pc.row_words = (uint32_t)(w->gpu_row_bytes / 4);
	pc.K = (uint32_t)w->K;
	pc.N = (uint32_t)w->N;
	pc.x_off = xq;
	pc.y_off = y;
	seq_pipe(w->type);
	vkCmdPushConstants(V.cb, V.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
	vkCmdDispatch(V.cb, (pc.N + V.rows_per_wg - 1) / V.rows_per_wg, 1, 1);
}

static void
seq_op(uint32_t pipe, const uint32_t *p, uint32_t groups)
{
	seq_pipe(pipe);
	vkCmdPushConstants(V.cb, V.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64, p);
	vkCmdDispatch(V.cb, groups, 1, 1);
}

static uint32_t
fbits(float f)
{
	uint32_t u;
	memcpy(&u, &f, 4);
	return u;
}

/* v = a [+ s*b | silu(a)*b]; x_out <- v; v = rmsnorm(v)*w; f_out <- v;
 * q_out <- q8k(v).  Unused offsets are NONE. */
static void
seq_norm(uint32_t a, uint32_t b, float s, uint32_t n, uint32_t x_out, uint32_t w,
    float eps, uint32_t f_out, uint32_t q_out, uint32_t mode)
{
	uint32_t p[16];
	memset(p, 0, sizeof(p));
	p[0] = a; p[1] = b; p[2] = fbits(s); p[3] = n; p[4] = x_out; p[5] = w;
	p[6] = fbits(eps); p[7] = f_out; p[8] = q_out; p[9] = mode;
	seq_op(PIPE_NORM, p, 1);
}

static void
seq_get(uint32_t off, uint32_t n, float *dst)
{
	V.dl[V.ndl].off = off;
	V.dl[V.ndl].n = n;
	V.dl[V.ndl].dst = dst;
	V.ndl++;
}

static void
seq_submit(void)
{
	VkSubmitInfo si;
	double t0 = vnow();
	int i;

	if (V.ndl) {
		VkBufferCopy rg[BMAX];
		barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
		    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
		for (i = 0; i < V.ndl; i++) {
			rg[i].srcOffset = rg[i].dstOffset = (VkDeviceSize)V.dl[i].off * 4;
			rg[i].size = (VkDeviceSize)V.dl[i].n * 4;
		}
		vkCmdCopyBuffer(V.cb, V.abuf, V.hbuf, (uint32_t)V.ndl, rg);
		barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
		    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
	}
	vkEndCommandBuffer(V.cb);
	memset(&si, 0, sizeof(si));
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &V.cb;
	if (vkQueueSubmit(V.q, 1, &si, V.fence) != VK_SUCCESS) {
		fprintf(stderr, "vulkan: submit failed\n");
		exit(1);
	}
	V.inflight = 1;
	vk_n_sub++;
	vk_t_sub += vnow() - t0;
}

void
vk_wait(void)
{
	int i;
	double t0 = vnow();
	if (!V.inflight)
		return;
	/* poll first: a blocking wait lets the core sleep and clock down */
	while (vkGetFenceStatus(V.dev, V.fence) == VK_NOT_READY && vnow() - t0 < 0.005)
		;
	if (vkWaitForFences(V.dev, 1, &V.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
		fprintf(stderr, "vulkan: device lost\n");
		exit(1);
	}
	vk_t_wait += vnow() - t0;
	vkResetFences(V.dev, 1, &V.fence);
	V.inflight = 0;
	for (i = 0; i < V.ndl; i++)
		memcpy(V.dl[i].dst, V.hmap + V.dl[i].off, sizeof(float) * V.dl[i].n);
	V.ndl = 0;
}

/* ---------------------------------------------------------------- */
/* plain matvec batches                                             */

void
vk_begin(void)
{
	V.nx = V.nd = 0;
}

int
vk_add(const wt *w, int e, const float *x, const q8k *xq, float *y)
{
	if (!w->gpu_n || w->gslot[e] < 0 || V.nd >= BMAX)
		return -1;
	V.d[V.nd].w = w;
	V.d[V.nd].e = e;
	V.d[V.nd].x = x;
	V.d[V.nd].xq = xq;
	V.d[V.nd].y = y;
	V.nd++;
	return 0;
}

void
vk_submit(void)
{
	uint32_t xo[BMAX], yo[BMAX];
	int i, k;

	if (V.nd == 0)
		return;
	seq_begin();
	for (i = 0; i < V.nd; i++) {
		for (k = 0; k < i; k++)
			if (V.d[k].x == V.d[i].x && V.d[k].w->K == V.d[i].w->K)
				break;
		xo[i] = k < i ? xo[k] : seq_put(V.d[i].xq, (uint32_t)V.d[i].w->K / 256 * Q8W);
	}
	seq_flush_up();
	for (i = 0; i < V.nd; i++) {
		yo[i] = seq_alloc((uint32_t)V.d[i].w->N);
		seq_matvec(V.d[i].w, V.d[i].e, xo[i], yo[i]);
		seq_get(yo[i], (uint32_t)V.d[i].w->N, V.d[i].y);
	}
	seq_submit();
}

/* ---------------------------------------------------------------- */
/* fused Mamba2 layer                                               */

int
vk_mamba_ok(int il)
{
	return V.ok && V.lay && il < V.nlay && V.lay[il].ok;
}

void
vk_state_reset(void)
{
	if (!V.ok || !V.sbuf)
		return;
	vk_wait();
	{
		VkCommandBufferBeginInfo bi;
		memset(&bi, 0, sizeof(bi));
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(V.cb, &bi);
		vkCmdFillBuffer(V.cb, V.sbuf, 0, VK_WHOLE_SIZE, 0);
		vkEndCommandBuffer(V.cb);
		submit_wait();
	}
}

/*
 * x (residual stream) in, x out after the mixer's residual add, and
 * xn = ffn_norm(x) for the MoE block; one submit:
 *   attn_norm+q8k, ssm_in, conv, scan, ssm_norm+q8k, ssm_out,
 *   residual + ffn_norm (+ q8k kept on the GPU for the MoE)
 */
void
vk_mamba_layer(const model *m, int il, float *x, float *xn)
{
	const hparams *h = &m->hp;
	const layer *L = &m->L[il];
	const struct lay *ly = &V.lay[il];
	const uint32_t E = (uint32_t)h->n_embd, DI = (uint32_t)h->d_inner;
	const uint32_t CH = (uint32_t)h->conv_ch, NH = (uint32_t)h->ssm_heads;
	uint32_t ax, q1, zx, xbc, dt, y, q2, o, p[16];

	seq_begin();
	if (V.prof) {
		vkCmdResetQueryPool(V.cb, V.qp, 0, 16);
		V.nts = 0;
	}
	ax = seq_put(x, E);
	seq_flush_up();
	ts();
	q1 = seq_alloc(E / 256 * Q8W);
	seq_norm(ax, NONE, 0, E, NONE, ly->an, h->eps, NONE, q1, 0);
	cbar();
	zx = seq_alloc((uint32_t)h->d_in_proj);
	seq_matvec(&L->ssm_in, 0, q1, zx);
	cbar();
	xbc = seq_alloc(CH);
	dt = seq_alloc(NH);
	memset(p, 0, sizeof(p));
	p[0] = zx + DI; p[1] = zx + DI + CH; p[2] = xbc; p[3] = dt; p[4] = ly->sconv;
	p[5] = ly->cw; p[6] = ly->cb; p[7] = ly->dtb; p[8] = CH; p[9] = NH;
	seq_op(PIPE_CONV, p, (CH + 255) / 256);
	cbar();
	y = seq_alloc(DI);
	memset(p, 0, sizeof(p));
	p[0] = xbc; p[1] = dt; p[2] = zx; p[3] = y; p[4] = ly->sssm; p[5] = ly->a; p[6] = ly->d;
	p[7] = DI; p[8] = (uint32_t)h->d_state; p[9] = (uint32_t)h->ssm_head_dim;
	seq_op(PIPE_SCAN, p, (DI + V.sg_per_wg - 1) / V.sg_per_wg);
	cbar();
	q2 = seq_alloc(DI / 256 * Q8W);
	seq_norm(y, NONE, 0, DI, NONE, ly->sn, h->eps, NONE, q2, 0);
	cbar();
	o = seq_alloc(E);
	seq_matvec(&L->ssm_out, 0, q2, o);
	cbar();
	V.xq_moe = XQ_RES;      /* outside the bump allocator: read by the MoE submit */
	V.x_out = seq_alloc(E);
	V.xn_out = seq_alloc(E);
	seq_norm(ax, o, h->res_scale, E, V.x_out, ly->fn, h->eps, V.xn_out, V.xq_moe, 0);
	seq_get(V.x_out, E, x);
	seq_get(V.xn_out, E, xn);
	ts();
	seq_submit();
	vk_wait();
	if (V.prof) {
		uint64_t t[16];
		int i;
		if (vkGetQueryPoolResults(V.dev, V.qp, 0, (uint32_t)V.nts, sizeof(t), t, 8,
		    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS) {
			for (i = 1; i < V.nts; i++)
				V.stage[i - 1] += (double)(t[i] - t[i - 1]) * V.ts_period * 1e-9;
			V.nstage++;
		}
	}
	(void)0;
}

void
vk_prof_report(void)
{
	static const char *nm[] = { "attn_norm+q", "ssm_in", "conv", "scan", "ssm_norm+q",
	    "ssm_out", "res+ffn_norm" };
	int i;
	if (!V.prof || !V.nstage)
		return;
	fprintf(stderr, "gpu mixer per layer (us):");
	for (i = 0; i < 7; i++)
		fprintf(stderr, " %s %.1f", nm[i], V.stage[i] / V.nstage * 1e6);
	fprintf(stderr, "\n");
}

/* ---------------------------------------------------------------- */
/* MoE: GPU-resident experts, while the CPU runs the others          */

void
vk_moe_begin(void)
{
	V.nmoe = 0;
}

int
vk_moe_add(const wt *gate, const wt *up, const wt *down, int e, float *out)
{
	if (V.nmoe >= 16 || !gate->gpu_n || gate->gslot[e] < 0 || !up->gpu_n ||
	    up->gslot[e] < 0 || !down->gpu_n || down->gslot[e] < 0)
		return -1;
	V.moe[V.nmoe].g = gate;
	V.moe[V.nmoe].u = up;
	V.moe[V.nmoe].d = down;
	V.moe[V.nmoe].e = e;
	V.moe[V.nmoe].out = out;
	V.nmoe++;
	return 0;
}

/* xq: the block input as q8k, or NULL to use the one the fused mixer
 * left on the GPU */
void
vk_moe_submit(const q8k *xq, int K)
{
	uint32_t xo, g[16], u[16], hq[16], o[16];
	int i;

	if (!V.nmoe)
		return;
	seq_begin();
	xo = xq ? seq_put(xq, (uint32_t)K / 256 * Q8W) : V.xq_moe;
	seq_flush_up();
	for (i = 0; i < V.nmoe; i++) {
		g[i] = seq_alloc((uint32_t)V.moe[i].g->N);
		u[i] = seq_alloc((uint32_t)V.moe[i].u->N);
		seq_matvec(V.moe[i].g, V.moe[i].e, xo, g[i]);
		seq_matvec(V.moe[i].u, V.moe[i].e, xo, u[i]);
	}
	cbar();
	for (i = 0; i < V.nmoe; i++) {
		uint32_t ff = (uint32_t)V.moe[i].g->N;
		hq[i] = seq_alloc(ff / 256 * Q8W);
		seq_norm(g[i], u[i], 0, ff, NONE, NONE, 0, NONE, hq[i], 1);
	}
	cbar();
	for (i = 0; i < V.nmoe; i++) {
		o[i] = seq_alloc((uint32_t)V.moe[i].d->N);
		seq_matvec(V.moe[i].d, V.moe[i].e, hq[i], o[i]);
		seq_get(o[i], (uint32_t)V.moe[i].d->N, V.moe[i].out);
	}
	seq_submit();
}
