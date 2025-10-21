#!/usr/bin/env python3
import sys
import re

# Mapper reads CM4IR logs from stdin and emits key-value pairs per metric
# Expected log line formats (from guided_diffusion/diffusion.py):
#   "img_ind: %d, PSNR: %.2f, LPIPS: %.4f"
#   "Total Average PSNR: %.2f"
#   "Total Average LPIPS: %.4f"

psnr_re = re.compile(r"PSNR:\s*([0-9]+\.?[0-9]*)")
lpips_re = re.compile(r"LPIPS:\s*([0-9]+\.?[0-9]*)")

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    # Per-image line
    m1 = psnr_re.search(line)
    m2 = lpips_re.search(line)
    if m1 is not None and m2 is not None and 'img_ind' in line:
        try:
            psnr_val = float(m1.group(1))
            lpips_val = float(m2.group(1))
            # Emit normalized keys so reducer can aggregate
            print(f"PSNR\t{psnr_val}")
            print(f"LPIPS\t{lpips_val}")
        except Exception:
            continue
    # Final summary lines (emit too; reducer will still average correctly)
    elif line.startswith('Total Average PSNR:'):
        try:
            psnr_val = float(line.split(':', 1)[1].strip())
            print(f"PSNR\t{psnr_val}")
        except Exception:
            pass
    elif line.startswith('Total Average LPIPS:'):
        try:
            lpips_val = float(line.split(':', 1)[1].strip())
            print(f"LPIPS\t{lpips_val}")
        except Exception:
            pass
