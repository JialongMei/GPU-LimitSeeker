# Third-party notices

GPU LimitSeeker's original Python implementation is MIT licensed. Windows/NVIDIA adapters implement local NVAPI/NVML/MACM interoperability; no NVIDIA DLL, MSI Afterburner executable, proprietary SDK source, driver or stress-test executable is included in this repository or wheel. Users obtain and license their installed drivers and Afterburner separately. This project is not affiliated with NVIDIA, MSI or the upstream stress-tool authors.

## GPU Auto Optimizer (GAO)

External pinned tool: [Rovey/gpu-auto-optimizer v0.4.9](https://github.com/Rovey/gpu-auto-optimizer/releases/tag/v0.4.9), source commit `638de2a8b20fd9a2e9f15eea74a7029a86cb59b0`.

The modified Python exact-vector bridge in `hardware/raw_curve.py` adapts ABI/layout definitions from upstream [`src/hw/nvapi.cpp`](https://github.com/Rovey/gpu-auto-optimizer/blob/638de2a8b20fd9a2e9f15eea74a7029a86cb59b0/src/hw/nvapi.cpp) and [`src/core/vf_curve.hpp`](https://github.com/Rovey/gpu-auto-optimizer/blob/638de2a8b20fd9a2e9f15eea74a7029a86cb59b0/src/core/vf_curve.hpp), under the MIT notice below. It is a derived interoperability binding, not the unmodified upstream implementation; no NVIDIA SDK header or DLL is bundled.

- Archive: `GpuAutoOptimizer-v0.4.9-win-x64.zip`
- Archive SHA-256: `ec276d6c870cbd225d43d90591ab1b5e829d05925503c4c71a4d92de2ddcccb8`
- `gao.exe` SHA-256: `f2390c2113329e8a72682fddca377d6f44a51d019f4f3c3366c45fcaba45ee7e`
- [Upstream license](https://github.com/Rovey/gpu-auto-optimizer/blob/638de2a8b20fd9a2e9f15eea74a7029a86cb59b0/LICENSE):

```text
MIT License

Copyright (c) 2026 Rovey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The optional downloader preserves GAO's own `LICENSE` and `THIRD-PARTY-NOTICES.md` from the pinned archive beside the executable. GAO has its own upstream dependencies; its notice file remains authoritative for those components. GPU LimitSeeker neither copies nor replaces GAO's optimizer: it invokes guarded control/benchmark operations and independently evaluates evidence.

## memtest_vulkan

External pinned tool: [GpuZelenograd/memtest_vulkan v0.5.0](https://github.com/GpuZelenograd/memtest_vulkan/releases/tag/v0.5.0).

- Binary: `memtest_vulkan-v0.5.0.exe`
- SHA-256: `09e67704210762ae8d8ad70fe6d71275eefbe815cc2c0658cd5afc8e231d0df6`
- [Upstream license](https://github.com/GpuZelenograd/memtest_vulkan/blob/v0.5.0/LICENSE):

```text
zlib License

Copyright (c) 2022 galkinvv by GpuZelenograd

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
```

Hashes identify pinned bytes; they are not a security audit or an Authenticode signature. Downloads are explicit and never executed automatically. Do not bypass antivirus, SmartScreen, UAC or execution policy to use this tool.
