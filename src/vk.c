#include "vk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vulkan/vulkan.h>

#include "gguf.h"
#include "shaders.h"

#define XCAP (64u * 1024u)       /* words of q8k input blocks per batch */
#define Q8W ((uint32_t)(sizeof(q8k) / 4))
#define YCAP (256u * 1024u)      /* floats of outputs per batch */
#define BMAX 64
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

	VkBuffer wbuf, abuf, hbuf, ubuf;
	VkDeviceMemory wmem, amem, hmem, umem;
	size_t wcap, wused;
	float *hmap;
	uint8_t *umap;

	VkDescriptorSetLayout dsl;
	VkPipelineLayout pl;
	VkDescriptorPool dp;
	VkDescriptorSet ds;
	VkPipeline pipe[16];
	VkCommandPool cp;
	VkCommandBuffer cb;
	VkFence fence;
	uint32_t rows_per_wg;
	int dot;                /* native packed int8 dot product */

	/* current batch */
	int nx, nd;
	uint32_t xused, yused;
	struct { const float *x; uint32_t off, K; } xs[BMAX];  /* off in words */
	struct { float *y; uint32_t type; pushc pc; } d[BMAX];
	int inflight;
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
vk_init(size_t budget, int devidx, int verbose)
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
	VkDescriptorSetLayoutBinding lb[3];
	VkDescriptorSetLayoutCreateInfo lci;
	VkPushConstantRange pr;
	VkPipelineLayoutCreateInfo plci;
	VkDescriptorPoolSize ps;
	VkDescriptorPoolCreateInfo dpci;
	VkDescriptorSetAllocateInfo dsai;
	VkDescriptorBufferInfo dbi[3];
	VkWriteDescriptorSet wr[3];
	VkCommandPoolCreateInfo cpci;
	VkCommandBufferAllocateInfo cbai;
	VkFenceCreateInfo fci;
	VkPhysicalDeviceShaderIntegerDotProductFeatures idf;
	const char *dotext = VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
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
	    sp.subgroupSize == 0 || 128 % sp.subgroupSize) {
		fprintf(stderr, "vulkan: %s lacks subgroup arithmetic\n", V.name);
		return -1;
	}
	V.rows_per_wg = 128 / sp.subgroupSize;

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
			for (i = 0; i < ne; i++)
				if (!strcmp(ep[i].extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
					V.dot = 1;
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
		dci.enabledExtensionCount = 1;
		dci.ppEnabledExtensionNames = &dotext;
	}
	CK(vkCreateDevice(V.pd, &dci, NULL, &V.dev));
	vkGetDeviceQueue(V.dev, V.qf, 0, &V.q);

	V.wcap = budget;
	if (V.wcap < 4096)
		V.wcap = 4096;
	if (mkbuf(V.wcap, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	    0, &V.wbuf, &V.wmem, NULL))
		return -1;
	if (mkbuf((XCAP + YCAP) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
	    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	    0, &V.abuf, &V.amem, NULL))
		return -1;
	if (mkbuf((XCAP + YCAP) * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	    VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 1, &V.hbuf, &V.hmem, &p))
		return -1;
	V.hmap = p;

	memset(lb, 0, sizeof(lb));
	for (i = 0; i < 3; i++) {
		lb[i].binding = i;
		lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		lb[i].descriptorCount = 1;
		lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	memset(&lci, 0, sizeof(lci));
	lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	lci.bindingCount = 3;
	lci.pBindings = lb;
	CK(vkCreateDescriptorSetLayout(V.dev, &lci, NULL, &V.dsl));
	pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pr.offset = 0;
	pr.size = sizeof(pushc);
	memset(&plci, 0, sizeof(plci));
	plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plci.setLayoutCount = 1;
	plci.pSetLayouts = &V.dsl;
	plci.pushConstantRangeCount = 1;
	plci.pPushConstantRanges = &pr;
	CK(vkCreatePipelineLayout(V.dev, &plci, NULL, &V.pl));
	ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	ps.descriptorCount = 3;
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
	for (i = 0; i < 3; i++) {
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
	vkUpdateDescriptorSets(V.dev, 3, wr, 0, NULL);

#define PIPE(t, a) (V.dot ? mkpipe(t, a, sizeof(a)) : mkpipe(t, a##_nd, sizeof(a##_nd)))
	if (PIPE(GT_Q2_K, spv_q2_k) || PIPE(GT_Q3_K, spv_q3_k) ||
	    PIPE(GT_Q4_K, spv_q4_k) || PIPE(GT_Q5_K, spv_q5_k) ||
	    PIPE(GT_Q6_K, spv_q6_k) || PIPE(GT_Q8_0, spv_q8_0))
		return -1;
#undef PIPE

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
	for (i = 0; i < 16; i++)
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
gpu_bytes(const wt *w, int rows)
{
	size_t sb = ggml_type_bytes(w->type), db = gpu_block_bytes(w->type);
	return (size_t)rows * (w->row_bytes / sb * db) * (size_t)w->E;
}

int
vk_upload(wt *w, int rows)
{
	size_t sb, db, rb, grb, fill = 0, total;
	uint64_t base = V.wused, flushed = V.wused;
	int e, r;
	void *p;

	if (!V.ok || !V.pipe[w->type] || rows <= 0)
		return -1;
	if (rows > w->N)
		rows = w->N;
	sb = ggml_type_bytes(w->type);
	db = gpu_block_bytes(w->type);
	rb = w->row_bytes;
	grb = rb / sb * db;
	total = gpu_bytes(w, rows);
	if (V.wused + total > V.wcap)
		return -1;
	if (!V.ubuf) {
		if (mkbuf(UPCHUNK, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 1, &V.ubuf, &V.umem, &p))
			return -1;
		V.umap = p;
	}
	/* rows [0, rows) of each expert, packed expert after expert */
	for (e = 0; e < w->E; e++)
		for (r = 0; r < rows; r++) {
			if (fill + grb > UPCHUNK) {
				if (flush_upload(fill, flushed))
					return -1;
				flushed += fill;
				fill = 0;
			}
			repack(w->type, w->data + ((size_t)e * (size_t)w->N + (size_t)r) * rb,
			    V.umap + fill, rb / sb);
			fill += grb;
		}
	if (flush_upload(fill, flushed))
		return -1;
	w->gpu_rows = rows;
	w->gpu_off = base;
	w->gpu_row_bytes = grb;
	V.wused += (total + 255) & ~(size_t)255;
	return 0;
}

/* round down to whole workgroups of rows */
static int
round_rows(int rows, int N)
{
	if (rows >= N)
		return N;
	rows -= rows % (int)V.rows_per_wg;
	return rows < 0 ? 0 : rows;
}

/*
 * Every batch runs its GPU rows and CPU rows concurrently, so each
 * dense matrix outside the MoE block gets the same GPU share (balanced
 * per batch).  Shared experts go wholly to the GPU: they are batched
 * with the routed experts, which only get what is left, spread evenly
 * over all experts' leading rows.  Everything is scaled down if even
 * the dense part does not fit.
 */
void
vk_plan_and_upload(model *m, int verbose)
{
	size_t dense = 0, shexp = 0, experts = 0, cap = V.wcap, left;
	double share = g_gpu_share, fe, fs = 1;
	double hot = 0, hot_gpu = 0;
	int i, ng = 0;

	for (i = 0; i < m->n_all; i++) {
		wt *w = m->all[i];
		size_t b = gpu_bytes(w, w->N);
		if (w->heat < 1)
			experts += b;
		else if (w->moe)
			shexp += b;
		else
			dense += b;
	}
	/* 256-byte alignment slack per tensor */
	cap -= cap < (size_t)m->n_all * 256 ? cap : (size_t)m->n_all * 256;
	if (share * dense > cap) {
		share = (double)cap / dense;
		left = 0;
	} else
		left = cap - (size_t)(share * dense);
	if (shexp > left) {
		fs = shexp ? (double)left / shexp : 0;
		left = 0;
	} else
		left -= shexp;
	fe = experts ? (double)left / experts : 0;
	for (i = 0; i < m->n_all; i++) {
		wt *w = m->all[i];
		double f = w->heat < 1 ? fe : w->moe ? fs : share;
		int rows = round_rows((int)(f * w->N), w->N);
		if (rows > 0 && vk_upload(w, rows) == 0)
			ng++;
		/* bytes read per token, total and on the GPU */
		hot += w->heat * (double)w->nbytes;
		hot_gpu += w->heat * (double)w->nbytes * w->gpu_rows / w->N;
	}
	if (verbose)
		fprintf(stderr, "vulkan: gpu rows: %.0f%% of dense, %.0f%% of shared experts, "
		    "%.1f%% of routed experts; %d/%d tensors, %.1f MiB; "
		    "%.0f%% of per-token weight bytes (%.0f of %.0f MiB)\n",
		    100 * share, 100 * fs, 100 * fe, ng, m->n_all, V.wused / 1048576.0,
		    100.0 * hot_gpu / hot, hot_gpu / 1048576.0, hot / 1048576.0);
}

/* ---------------------------------------------------------------- */
/* batches                                                          */

void
vk_begin(void)
{
	V.nx = V.nd = 0;
	V.xused = V.yused = 0;
}

int
vk_add(const wt *w, int e, const float *x, const q8k *xq, float *y)
{
	int i;
	uint32_t xo, nw = (uint32_t)w->K / 256 * Q8W;

	if (!w->gpu_rows || V.nd >= BMAX || V.yused + (uint32_t)w->gpu_rows > YCAP)
		return -1;
	for (i = 0; i < V.nx; i++)
		if (V.xs[i].x == x && V.xs[i].K == (uint32_t)w->K)
			break;
	if (i == V.nx) {
		if (V.nx >= BMAX || V.xused + nw > XCAP)
			return -1;
		V.xs[i].x = x;
		V.xs[i].K = (uint32_t)w->K;
		V.xs[i].off = V.xused;
		memcpy(V.hmap + V.xused, xq, (size_t)nw * 4);
		V.xused += nw;
		V.nx++;
	}
	xo = V.xs[i].off;
	V.d[V.nd].y = y;
	V.d[V.nd].type = w->type;
	V.d[V.nd].pc.w_off = (uint32_t)((w->gpu_off + (uint64_t)e * (uint64_t)w->gpu_rows *
	    w->gpu_row_bytes) / 4);
	V.d[V.nd].pc.row_words = (uint32_t)(w->gpu_row_bytes / 4);
	V.d[V.nd].pc.K = (uint32_t)w->K;
	V.d[V.nd].pc.N = (uint32_t)w->gpu_rows;
	V.d[V.nd].pc.x_off = xo;
	V.d[V.nd].pc.y_off = XCAP + V.yused;
	V.yused += ((uint32_t)w->gpu_rows + 3) & ~3u;
	V.nd++;
	return 0;
}

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

void
vk_submit(void)
{
	VkCommandBufferBeginInfo bi;
	VkBufferCopy rg;
	VkSubmitInfo si;
	uint32_t cur = UINT32_MAX;
	int i;

	double t0 = vnow();
	if (V.nd == 0)
		return;
	vk_n_sub++;
	memset(&bi, 0, sizeof(bi));
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(V.cb, &bi);
	rg.srcOffset = rg.dstOffset = 0;
	rg.size = (VkDeviceSize)V.xused * 4;
	vkCmdCopyBuffer(V.cb, V.hbuf, V.abuf, 1, &rg);
	barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
	    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
	vkCmdBindDescriptorSets(V.cb, VK_PIPELINE_BIND_POINT_COMPUTE, V.pl, 0, 1, &V.ds, 0, NULL);
	for (i = 0; i < V.nd; i++) {
		if (V.d[i].type != cur) {
			cur = V.d[i].type;
			vkCmdBindPipeline(V.cb, VK_PIPELINE_BIND_POINT_COMPUTE, V.pipe[cur]);
		}
		vkCmdPushConstants(V.cb, V.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushc), &V.d[i].pc);
		vkCmdDispatch(V.cb, (V.d[i].pc.N + V.rows_per_wg - 1) / V.rows_per_wg, 1, 1);
	}
	barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
	    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	rg.srcOffset = rg.dstOffset = (VkDeviceSize)XCAP * 4;
	rg.size = (VkDeviceSize)V.yused * 4;
	vkCmdCopyBuffer(V.cb, V.abuf, V.hbuf, 1, &rg);
	barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
	    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
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
	for (i = 0; i < V.nd; i++)
		memcpy(V.d[i].y, V.hmap + V.d[i].pc.y_off, sizeof(float) * V.d[i].pc.N);
}
