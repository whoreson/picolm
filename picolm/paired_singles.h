/* paired_singles.h -- PowerPC 750CL / Broadway paired-singles as assembler
 * macros, so they work with a binutils that has no ps_* mnemonics.
 *
 * STATUS: EXPERIMENTAL, OFF BY DEFAULT. Read "TARGET STATUS" before enabling.
 *
 * WHY THIS EXISTS
 * ---------------
 * The Wii's stock binutils (GNU as 2.18.0.20080103, Debian) has no
 * paired-singles support at all: no -mpaired switch, and the mnemonics are
 * rejected under every -m option tried (-many, -mcell, -m7400, -mppc,
 * -maltivec, -mcom, -me500), all failing with
 * "Unrecognized opcode: ps_add". objdump 2.18 cannot disassemble them either.
 * So inline asm written as "ps_add 1,2,3" cannot be assembled on that machine
 * no matter how the surrounding C is structured.
 *
 * GNU as *macros* ARE supported, and the assembler can do integer arithmetic
 * on the register numbers GCC substitutes. So we define each ps_* instruction
 * once as a .macro that emits the equivalent .long, and every call site keeps
 * reading like ordinary paired-singles assembly.
 *
 *     .macro ps_add rt, ra, rb
 *         .long ((4<<26)|((\rt)<<21)|((\ra)<<16)|((\rb)<<11)|(21<<1))
 *     .endm
 *
 * Verified on WiiLinux gcc 4.3.2 / as 2.18: `ps_add_m 0, 0, 12` emits exactly
 * 0x1000602a, which matches the field-by-field reference encoder.
 *
 * ENCODING REFERENCE (PowerPC ISA 2.00/2.01, 750CL GQR paired singles)
 *
 *   A-form (primary opcode 4): ps_add/ps_sub/ps_mul/ps_madd/ps_madds0/1,
 *                              ps_sum0/1, and psq_lx/psq_stx
 *     0..5   PO=4      6..10 FRT    11..15 FRA
 *     16..20 FRB      21..25 FRC    26..30 XO (5 bit)   31 Rc=0
 *
 *   X-form (primary opcode 4): ps_mr/ps_swap/ps_mergeXX/ps_neg/ps_abs
 *     0..5   PO=4      6..10 FRT    11..15 FRA
 *     16..20 FRB      21..30 XO (10 bit)                31 Rc=0
 *
 *   psq_lx / psq_stx: X-form with W in bit 20 (from MSB's XO field) and I in
 *   bit 19; they are encoded here with their own XO (6 / 7).
 *
 *   D-form psq_l (PO=56) / psq_st (PO=57):
 *     0..5   PO        6..10 FRT       11..15 RA (base GPR, 0 = zero)
 *     16..27 DQ (displacement, multiple of 4)
 *     28..29 W (0=single, 1=paired)    30..31 I (GQR index)
 *
 * TARGET STATUS -- WHY THIS IS OFF BY DEFAULT
 * -------------------------------------------
 * On the current WiiLinux install these instructions trap with SIGILL for
 * both root and userspace, regardless of encoding correctness:
 *
 *     /proc/cpuinfo   : cpu : 750CL        (paired-singles hardware present)
 *     /proc/config.gz : # CONFIG_ALTIVEC is not set
 *
 * PPC kernel support for AltiVec/paired-singles state -- including setting
 * HID2[LSQE], which every ps_* instruction requires -- is gated behind
 * CONFIG_ALTIVEC, so with it disabled each ps_* instruction raises SIGILL.
 * Confirmed both by the missing mnemonics and by hand-encoded words: an
 * encoded ps_add and even a plain mfspr of HID2 both trap.
 *
 * Therefore enabling PICOLM_PS on such a kernel gives a binary that dies with
 * "Illegal instruction" on first use -- not a slow binary. Defining
 * PICOLM_PS_LONG_OPCODES only removes the *assembler* obstacle; it does not
 * make the instructions executable. Both are opt-in and off by default.
 *
 * WHEN THIS IS WORTH USING
 * ------------------------
 *  - A kernel built with CONFIG_ALTIVEC=y, on a toolchain without ps_*
 *    mnemonics (exactly the Wii's as 2.18 situation).
 *  - Producing an object for either of the above without changing binutils.
 */

#ifndef PICOLM_PAIRED_SINGLES_H
#define PICOLM_PAIRED_SINGLES_H

#ifdef PICOLM_PS_LONG_OPCODES

/*
 * Emit this ONCE per translation unit, before any use, via
 *     PICOLM_PS_ASM_PRELUDE
 * inside an __asm__ block (see quant.c). It defines the macros only; it emits
 * no code.
 *
 * Only the instructions actually used by PicoLM are defined. Adding more is
 * mechanical: look up the form (A/X/D), then copy the field layout above.
 */
#define PICOLM_PS_ASM_PRELUDE                                            \
    /* ---- arithmetic, A-form ---- */                                   \
    ".macro ps_add rt, ra, rb\n"                                         \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|((\\rb)<<11)|(21<<1))\n" \
    ".endm\n"                                                            \
    ".macro ps_sub rt, ra, rb\n"                                         \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|((\\rb)<<11)|(20<<1))\n" \
    ".endm\n"                                                            \
    /* ps_madd rt, ra, rb, rc  ->  rt = ra*rb + rc */                    \
    ".macro ps_madd rt, ra, rb, rc\n"                                    \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|((\\rb)<<11)|((\\rc)<<6)|(29<<1))\n" \
    ".endm\n"                                                            \
    /* ps_madds0/s1 rt, ra, rb, rc -> rt = ra*[rb.lo|rb.hi] + rc */      \
    ".macro ps_madds0 rt, ra, rb, rc\n"                                  \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|((\\rb)<<11)|((\\rc)<<6)|(14<<1))\n" \
    ".endm\n"                                                            \
    ".macro ps_madds1 rt, ra, rb, rc\n"                                  \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|((\\rb)<<11)|((\\rc)<<6)|(15<<1))\n" \
    ".endm\n"                                                            \
    /* ---- X-form ---- */                                               \
    /* ps_mr rt, ra  -> rt = ra */                                       \
    ".macro ps_mr rt, ra\n"                                              \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|(0<<11)|(72<<1))\n"      \
    ".endm\n"                                                            \
    /* ps_swap rt, ra -> swap the two floats in ra into rt */            \
    ".macro ps_swap rt, ra\n"                                            \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|(0<<11)|(84<<1))\n"      \
    ".endm\n"                                                            \
    /* ps_merge00/11 rt, ra, rb -> interleave high/low halves */         \
    ".macro ps_merge00 rt, ra, rb\n"                                     \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|((\\rb)<<11)|(528<<1))\n" \
    ".endm\n"                                                            \
    ".macro ps_merge11 rt, ra, rb\n"                                     \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|((\\rb)<<11)|(624<<1))\n" \
    ".endm\n"                                                            \
    /* ---- loads / stores ---- */                                       \
    /* psq_l rt, dq(ra), W, I   (dq must be a multiple of 4) */          \
    ".macro psq_l rt, dq, ra, w, i\n"                                    \
    "  .long ((56<<26)|((\\rt)<<21)|((\\ra)<<16)|(((\\dq)&0xfff)<<4)|(((\\w)&1)<<1)|((\\i)&1))\n" \
    ".endm\n"                                                            \
    ".macro psq_st rt, dq, ra, w, i\n"                                   \
    "  .long ((57<<26)|((\\rt)<<21)|((\\ra)<<16)|(((\\dq)&0xfff)<<4)|(((\\w)&1)<<1)|((\\i)&1))\n" \
    ".endm\n"                                                            \
    /* psq_lx rt, ra, rb, W, I  (address = ra + rb) */                   \
    ".macro psq_lx rt, ra, rb, w, i\n"                                   \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|((\\rb)<<11)|(((\\w)&1)<<10)|(((\\i)&1)<<9)|(6<<1))\n" \
    ".endm\n"                                                            \
    ".macro psq_stx rt, ra, rb, w, i\n"                                  \
    "  .long ((4<<26)|((\\rt)<<21)|((\\ra)<<16)|((\\rb)<<11)|(((\\w)&1)<<10)|(((\\i)&1)<<9)|(7<<1))\n" \
    ".endm\n"

#else /* !PICOLM_PS_LONG_OPCODES */

/*
 * Native-mnemonic path: the assembler understands ps_* itself, but PicoLM's
 * call sites pass psq_l/psq_st as (rt, dq, ra, w, i) -- displacement and base
 * register separately -- because that is what the .long form needs. Map that
 * back to the canonical "dq(ra)" syntax so one set of call sites serves both.
 */
#define PICOLM_PS_ASM_PRELUDE                                            \
    ".macro psq_l rt, dq, ra, w, i\n"                                   \
    "  psq_l \\rt, \\dq(\\ra), \\w, \\i\n"                     \
    ".endm\n"                                                            \
    ".macro psq_st rt, dq, ra, w, i\n"                                  \
    "  psq_st \\rt, \\dq(\\ra), \\w, \\i\n"                     \
    ".endm\n"

#endif /* PICOLM_PS_LONG_OPCODES */

#endif /* PICOLM_PAIRED_SINGLES_H */
