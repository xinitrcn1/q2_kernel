/*
 * dec(): a lane's 16 weights of a 256-block as four groups of four,
 *   w = D * sa * q + M * sb   (q packed signed bytes, per group)
 * for q2k's transcoded types and IQ4_XS, in the word-aligned layouts
 * the upload repacks them into (vk.c repack()):
 *   X2  116 B: lv[64] (Q2_K order) sg[32] (Q3_K hmask order) mult[16] d
 *   X3  140 B: q[128] (4-bit level|sign<<3, IQ4_XS order) mult[8] d
 *   X3S 140 B: as X3
 *   X1   76 B: lv[64] (g+1, Q2_K order) sd[8] (mult | dneg<<7) d
 *   IQ4_XS 136 B: d sh sl[4] qs[128]
 * Included by matvec.comp and mmq.comp after W, ONES and b8().
 */
#ifndef GRP_DEFINED
#define GRP_DEFINED
struct grp {
	uint q[4];      /* packed weights */
	uint o[4];      /* element offset in the block */
	int sa[4], sb[4];
	float D, M;
};
#endif

/* negate the bytes of w whose bit in s (0x01 per byte) is set; every
 * byte of w is in 1..127, so no byte carries into the next */
uint negb(uint w, uint s)
{
	return (w ^ (s * 255u)) + s;
}

uint wb(uint bw, uint i) { return (W[bw + (i >> 2)] >> (8u * (i & 3u))) & 255u; }

#if QT == 100 /* X2: d/8 * mult * {8,25,43}[lv] * sign */
#define BW 29u
void dec(uint bw, uint l, out grp g)
{
	uint c = l >> 3, o = (4u * l) & 31u, h = o >> 4;
	uint q = W[bw + l], hm = W[bw + 16u + (o >> 2)];
	[[unroll]] for (uint j = 0u; j < 4u; j++) {
		uint lv = (q >> (2u * j)) & 0x03030303u;
		uint mag = 0x08080808u + lv * 17u + ((lv >> 1) & ONES);
		g.q[j] = negb(mag, (hm >> (4u * c + j)) & ONES);
		g.o[j] = 128u * c + 32u * j + o;
		g.sa[j] = int(wb(bw, 96u + 8u * c + 2u * j + h));
		g.sb[j] = 0;
	}
	g.D = unpackHalf2x16(W[bw + 28u]).x * 0.125;
	g.M = 0.0;
}
#elif QT == 101 || QT == 102 /* X3: d/4 * mult * +-{4,..,52,62}; X3S: d * mult * +-{1,..,15} */
#define BW 35u
void dec(uint bw, uint l, out grp g)
{
	uint ib = l >> 1, hf = l & 1u;
	int sc = int(wb(bw, 128u + ib));
	[[unroll]] for (uint k = 0u; k < 2u; k++) {
		uint w = W[bw + 4u * ib + 2u * hf + k];
		[[unroll]] for (uint n = 0u; n < 2u; n++) {
			uint code = (w >> (4u * n)) & 0x0f0f0f0fu;
			uint lv = code & 0x07070707u;
#if QT == 101
			uint mag = lv * 8u + 0x04040404u + (((lv + ONES) >> 3) & ONES) * 2u;
#else
			uint mag = lv * 2u + ONES;
#endif
			uint gi = 2u * k + n;
			g.q[gi] = negb(mag, (code >> 3) & ONES);
			g.o[gi] = 32u * ib + 16u * n + 8u * hf + 4u * k;
			g.sa[gi] = sc;
			g.sb[gi] = 0;
		}
	}
#if QT == 101
	g.D = unpackHalf2x16(W[bw + 34u]).x * 0.25;
#else
	g.D = unpackHalf2x16(W[bw + 34u]).x;
#endif
	g.M = 0.0;
}
#elif QT == 103 /* X1: d * mult * (g + delta), via (g+1) and the bsums */
#define BW 19u
void dec(uint bw, uint l, out grp g)
{
	uint c = l >> 3, o = (4u * l) & 31u;
	uint q = W[bw + l];
	float d = unpackHalf2x16(W[bw + 18u]).x;
	[[unroll]] for (uint j = 0u; j < 4u; j++) {
		uint sd = wb(bw, 64u + 4u * c + j);
		int mult = int(sd & 127u);
		g.q[j] = (q >> (2u * j)) & 0x03030303u;
		g.o[j] = 128u * c + 32u * j + o;
		g.sa[j] = mult;
		g.sb[j] = mult * ((sd & 128u) != 0u ? -9 : -7);
	}
	g.D = d;
	g.M = d * 0.125;
}
#elif QT == 23 /* IQ4_XS: d * (ls - 32) * kvalues[nibble] */
#define BW 34u
/* kvalues_iq4nl as bytes */
const uint KV[4] = uint[4](0xbfad9881u, 0xf6eaddcfu, 0x26190d01u, 0x71594535u);
uint kv4(uint nib)
{
	uint r = 0u;
	[[unroll]] for (uint i = 0u; i < 4u; i++) {
		uint n = (nib >> (8u * i)) & 15u;
		r |= ((KV[n >> 2] >> (8u * (n & 3u))) & 255u) << (8u * i);
	}
	return r;
}
void dec(uint bw, uint l, out grp g)
{
	uint ib = l >> 1, hf = l & 1u;
	uint w0 = W[bw];
	uint sh = w0 >> 16;
	int ls = int(((wb(bw, 4u + ib / 2u) >> (4u * (ib & 1u))) & 15u) | (((sh >> (2u * ib)) & 3u) << 4));
	[[unroll]] for (uint k = 0u; k < 2u; k++) {
		uint w = W[bw + 2u + 4u * ib + 2u * hf + k];
		[[unroll]] for (uint n = 0u; n < 2u; n++) {
			uint gi = 2u * k + n;
			g.q[gi] = kv4((w >> (4u * n)) & 0x0f0f0f0fu);
			g.o[gi] = 32u * ib + 16u * n + 8u * hf + 4u * k;
			g.sa[gi] = ls - 32;
			g.sb[gi] = 0;
		}
	}
	g.D = unpackHalf2x16(w0).x;
	g.M = 0.0;
}
#endif
