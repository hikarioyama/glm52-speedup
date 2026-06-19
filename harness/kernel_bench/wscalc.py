# working set sizes
q2_blk = 256//16 + 256//4 + 4   # scales + qs + dm = 16+64+4 = 84
q3_blk = 256//8 + 256//4 + 12 + 2  # hmask+qs+scales+d = 32+64+12+2 = 110
q8_blk = 4 + 256 + 16*2          # d + qs + bsums = 292
for name,blk,rows,nblk in [("Q2 n6144",q2_blk,4096,24),("Q2 n2048",q2_blk,8192,8),
                           ("Q3 n2048",q3_blk,8192,8),("Q3 n6144",q3_blk,4096,24)]:
    w = blk*rows*nblk
    a = q8_blk*nblk
    print(f"{name}: weights={w/1e6:.2f}MB act={a/1e3:.1f}KB total={(w+a)/1e6:.2f}MB")
