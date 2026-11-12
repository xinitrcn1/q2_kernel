# q2k: GGUF kernel for Granite 4.0-H (BSD make)

PROG=		q2k
SRCS=		main.c gguf.c quant.c kq.c pool.c model.c tok.c ucd.c vk.c vmath.c
HDRS=		gguf.h quant.h kq.h pool.h model.h tok.h vk.h vmath.h
QTYPES=		q2_k q3_k q4_k q5_k q6_k q8_0
QT.q2_k=	10
QT.q3_k=	11
QT.q4_k=	12
QT.q5_k=	13
QT.q6_k=	14
QT.q8_0=	8

CC?=		cc
OPT?=		-O2 -march=native
CFLAGS+=	-std=c99 ${OPT} -Wall -Wextra -pedantic -Isrc -Igen -I/usr/local/include
LDFLAGS+=	-L/usr/local/lib
LDLIBS+=	-lvulkan -lpthread -lm
GLSLANG?=	glslangValidator

OBJS=		${SRCS:%.c=build/%.o}
SPVH=		${QTYPES:%=gen/spv_%.h} ${QTYPES:%=gen/spv_%_nd.h}

all: ${PROG}

${PROG}: ${OBJS}
	${CC} ${LDFLAGS} -o ${.TARGET} ${OBJS} ${LDLIBS}

.for s in ${SRCS}
build/${s:R}.o: src/${s} ${HDRS:%=src/%}
	@mkdir -p build
	${CC} ${CFLAGS} -c -o ${.TARGET} src/${s}
.endfor

build/vk.o: gen/shaders.h

# per type: a DP4A variant and one with an emulated dot (_nd)
.for t in ${QTYPES}
gen/spv_${t}.h: shaders/matvec.comp
	@mkdir -p gen
	${GLSLANG} -V --target-env vulkan1.1 -DQT=${QT.${t}} --vn spv_${t} \
	    -o ${.TARGET} shaders/matvec.comp >/dev/null
gen/spv_${t}_nd.h: shaders/matvec.comp
	@mkdir -p gen
	${GLSLANG} -V --target-env vulkan1.1 -DQT=${QT.${t}} -DNO_DOT --vn spv_${t}_nd \
	    -o ${.TARGET} shaders/matvec.comp >/dev/null
.endfor

gen/shaders.h: ${SPVH}
	@mkdir -p gen
	( echo '#include <stdint.h>'; \
	  for f in ${SPVH:T}; do echo "#include \"$$f\""; done ) > ${.TARGET}

# regenerate the pre-tokenizer's Unicode table (needs python3; the
# output is committed so normal builds do not)
ucd:
	python3 tools/gen_ucd.py > src/ucd.c

clean:
	rm -rf build gen ${PROG}

.PHONY: all clean ucd
