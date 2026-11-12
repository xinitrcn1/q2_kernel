# q2k: GGUF kernel for Granite 4.0-H (BSD make)

PROG=		q2k
SRCS=		main.c gguf.c quant.c kq.c pool.c model.c tok.c ucd.c vk.c vmath.c prefill.c iqtab.c qwen35.c transcode.c
HDRS=		gguf.h quant.h kq.h pool.h model.h tok.h vk.h vmath.h iqtab.h
QTYPES=		q2_k q3_k q4_k q5_k q6_k q8_0 iq4_xs x2 x3 x3s x1
QT.q2_k=	10
QT.q3_k=	11
QT.q4_k=	12
QT.q5_k=	13
QT.q6_k=	14
QT.q8_0=	8
QT.iq4_xs=	23
QT.x2=		100
QT.x3=		101
QT.x3s=		102
QT.x1=		103

CC?=		cc
OPT?=		-O2 -march=native
CFLAGS+=	-std=c99 ${OPT} -Wall -Wextra -pedantic -Isrc -Igen -I/usr/local/include
LDFLAGS+=	-L/usr/local/lib
LDLIBS+=	-lvulkan -lpthread -lm
GLSLANG?=	glslangValidator

OBJS=		${SRCS:%.c=build/%.o}
SPVH=		${QTYPES:%=gen/spv_%.h} ${QTYPES:%=gen/spv_%_nd.h} \
		${QTYPES:%=gen/spv_%_b.h} ${QTYPES:%=gen/spv_%_ndb.h}

all: ${PROG}

${PROG}: ${OBJS}
	${CC} ${LDFLAGS} -o ${.TARGET} ${OBJS} ${LDLIBS}

.for s in ${SRCS}
build/${s:R}.o: src/${s} ${HDRS:%=src/%}
	@mkdir -p build
	${CC} ${CFLAGS} -c -o ${.TARGET} src/${s}
.endfor

build/vk.o: gen/shaders.h

# per type: DP4A and emulated-dot (_nd) variants of the matvec (decode)
# and of the tiled GEMM (_b, mmq.comp: batches)
.for t in ${QTYPES}
gen/spv_${t}.h: shaders/matvec.comp shaders/xdec.glsl
	@mkdir -p gen
	${GLSLANG} -V --target-env vulkan1.1 -Ishaders -DQT=${QT.${t}} --vn spv_${t} \
	    -o ${.TARGET} shaders/matvec.comp >/dev/null
gen/spv_${t}_nd.h: shaders/matvec.comp shaders/xdec.glsl
	@mkdir -p gen
	${GLSLANG} -V --target-env vulkan1.1 -Ishaders -DQT=${QT.${t}} -DNO_DOT --vn spv_${t}_nd \
	    -o ${.TARGET} shaders/matvec.comp >/dev/null
gen/spv_${t}_b.h: shaders/mmq.comp shaders/xdec.glsl
	@mkdir -p gen
	${GLSLANG} -V --target-env vulkan1.1 -Ishaders -DQT=${QT.${t}} --vn spv_${t}_b \
	    -o ${.TARGET} shaders/mmq.comp >/dev/null
gen/spv_${t}_ndb.h: shaders/mmq.comp shaders/xdec.glsl
	@mkdir -p gen
	${GLSLANG} -V --target-env vulkan1.1 -Ishaders -DQT=${QT.${t}} -DNO_DOT --vn spv_${t}_ndb \
	    -o ${.TARGET} shaders/mmq.comp >/dev/null
.endfor

OPS=		conv:2 scan:3 qblk:4 bw:5
.for o in ${OPS}
gen/spv_op_${o:C/:.*//}.h: shaders/ops.comp
	@mkdir -p gen
	${GLSLANG} -V --target-env vulkan1.1 -DOP=${o:C/.*://} --vn spv_op_${o:C/:.*//} \
	    -o ${.TARGET} shaders/ops.comp >/dev/null
SPVH+=		gen/spv_op_${o:C/:.*//}.h
.endfor

gen/shaders.h: ${SPVH}
	@mkdir -p gen
	( echo '#include <stdint.h>'; \
	  for f in ${SPVH:T}; do echo "#include \"$$f\""; done ) > ${.TARGET}

# regenerate the i-quant tables from ggml's ggml-common.h
iqtab:
	python3 tools/gen_iqtab.py ${GGML_COMMON:U$${HOME}/llama.cpp/ggml/src/ggml-common.h} > src/iqtab.c

# regenerate the pre-tokenizer's Unicode table (needs python3; the
# output is committed so normal builds do not)
ucd:
	python3 tools/gen_ucd.py > src/ucd.c

clean:
	rm -rf build gen ${PROG}

.PHONY: all clean ucd iqtab
