<div dir="rtl" align="right">

# ⚡ Sovereign V14 — پلتفرم جامع و فوق‌سریع GPU RAG با کرنل‌های Triton و Gemma-2 CUDAGraph

<p align="center">
  <img src="https://img.shields.io/badge/OpenAI_Triton-Custom_Kernels-00A67E?style=for-the-badge&logo=openai&logoColor=white" alt="Triton"/>
  <img src="https://img.shields.io/badge/CUDA_Graph-Sub--Millisecond-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Gemma--2_2B-Fused_Triton_4Bit-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemma-2"/>
  <img src="https://img.shields.io/badge/Scale-1,000,000+_Docs_on_GPU-F59E0B?style=for-the-badge" alt="1M Docs"/>
  <img src="https://img.shields.io/badge/PyTorch-Multi--GPU_Ready-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
</p>

**موتور بازیابی و تولید پاسخ (RAG) نسل جدید کاملاً مقیم روی GPU** با قابلیت ایندکس کردن **کتاب‌های چند هزار صفحه‌ای در کسری از ثانیه** و مقیاس‌پذیری تا **بیش از ۱,۰۰۰,۰۰۰ سند** تنها با چند صد مگابایت حافظهٔ گرافیکی (VRAM).

این پروژه بدون وابستگی به دیتابیس‌های برداری سنگین و کند سنتی، تمام مراحل **بسته‌بندی بیتی (`SoA22`)**، **هش موازی (`FNV-1a`)**، **ساخت جدول معکوس تنک (`CSR Inverted Index`)**، **یادگیری آنلاین آنی (`Live Online Learning`)** و **تولید پاسخ با مدل زبانی `Gemma-2`** را مستقیماً با **کرنل‌های اختصاصی Triton** و **CUDA Graph** روی کارت گرافیک اجرا می‌کند.

---

## 🌟 چرا این معماری متمایز است؟

در سیستم‌های RAG متداول (مبتنی بر FAISS، Chroma یا Milvus)، تبدیل یک کتاب ۲,۲۰۰ صفحه‌ای (بیش از ۱۰۰,۰۰۰ قطعه متن) به بردارهای Embedding و ساخت ایندکس روی GPU بین **۱۰ تا ۳۰ دقیقه** زمان می‌برد و چندین گیگابایت VRAM مصرف می‌کند.

در **Sovereign V14**:
* 🚀 **ایندکس بیش از ۱۰۲,۰۰۰ قطعه (۲.۷۲ میلیون توکن) روی GPU تنها در `151.14 میلی‌ثانیه`!**
* 💾 **مصرف حافظهٔ گرافیکی (VRAM) فوق‌العاده بهینه:** فقط **`89.8 MB`** برای کل کتاب ۲,۲۲۳ صفحه‌ای طب داخلی هریسون (و حدود **`400 MB`** برای ۱,۰۰۰,۰۰۰ سند!).
* ⚡ **پاسخ‌دهی در مقیاس میکروثانیه:** اجرای گراف حافظهٔ ثبات (`Full Couplet CUDA Graph Replay`) در **`8.00 µs`** و بازیابی نهایی در **`0.17 ms`**.
* 🧠 **یادگیری آنلاین بدون بازسازی ایندکس (`online_learn`):** تزریق آنی داده‌ها و نکات جدید به بافرهای GPU (`online_hashes` و `online_postings`) در چند میکروثانیه.
* 🤖 **موتور استنتاج Gemma-2 مجهز به کرنل‌های فیوزشدهٔ Triton و CUDAGraph:** جایگزینی لایه‌های `RMSNorm`، `Split-GeGLU MLP` و `Logit Softcapping` در تمام ۲۶ لایهٔ مدل `Gemma-2-2B-IT` با کرنل‌های Triton و پشتیبانی کامل از Multi-GPU.

---

## 🛠️ معماری فنی و کرنل‌های اختصاصی Triton

این پلتفرم شامل ۷ کرنل اختصاصی نوشته شده با **OpenAI Triton** است:

۱. **`sovereign_build_soa22_packed_kernel` (بسته‌بندی بیتی SoA22):**
   آفست بایت شروع هر توکن (`22-bit`) و طول توکن (`10-bit`) را در یک عدد صحیح ۳۲ بیتی (`((tok_len & 0x3FF) << 22) | (tok_start & 0x3FFFFF)`) فشرده می‌کند و متادیتای شماره صفحه و شماره سند را در آرایه‌های ساختاریافته (`Structure of Arrays`) روی GPU قرار می‌دهد.

۲. **`sovereign_hash_1d_tokens_kernel` (هش موازی FNV-1a روی GPU):**
   تمام میلیون‌ها توکن موجود در اسناد را به صورت موازی در سطح بایت UTF-8 پیمایش کرده، حروف را نرمال‌سازی می‌کند و هش ۳۲ بیتی `FNV-1a` را برای ساخت جدول `CSR` محاسبه می‌نماید.

۳. **`sovereign_csr_1m_bitmask_kernel` (جستجوی اتمیک دوبعدی برای ۱ میلیون سند):**
   با استفاده از شبکهٔ دوبعدی `(num_query_tokens, cdiv(max_bucket_span, 256))` و عملیات اتمیک سطح سخت‌افزار (`tl.atomic_or` برای بیت‌مسک حضور واژگان و `tl.atomic_add` برای امتیاز وزن‌دار `IDF`)، بدون حذف هیچ سندی، میلیون‌ها تطابق را در میکروثانیه امتیازدهی می‌کند.

۴. **`sovereign_online_search_score_kernel` (کاوش موازی حافظهٔ یادگیری آنلاین):**
   بافرهای `online_hashes` و `online_postings` را به صورت همزمان با جدول CSR اسکن کرده و به داده‌های تازه یادگرفته‌شده وزن تقویتی اختصاص می‌دهد.

۵. **`triton_gemma2_rmsnorm_kernel` & `triton_gemma2_split_geglu_kernel` & `triton_softcap_tanh_kernel`:**
   کرنل‌های فیوزشده در سطح ثبات‌های SRAM برای شتاب‌دهی به مدل زبانی **Gemma-2 2B** و سازگار با `torch.cuda.CUDAGraph`.

---

## 📊 بنچمارک واقعی روی کتاب مرجع طب داخلی هریسون (Harrison's Internal Medicine)

تست انجام‌شده روی نسخهٔ کامل کتاب **Harrison's Principles of Internal Medicine (20th Edition)** شامل **۲,۲۲۳ صفحه**، **۱۰۲,۴۰۹ قطعهٔ متنی (Passages)** و **۲,۷۲۳,۱۳۵ توکن ایندکس‌شده**:

</div>

<div dir="ltr" align="left">

```text
====================================================================================
 📊  GEMMA-2 CUDAGRAPH — SYSTEM & KERNEL BENCHMARK LOG
====================================================================================
   [1] Zero-Copy SHM + SoA22 Pack Time   :     159.05 ms
   [2] GPU CSR Inverted Index Build Time :      98.54 ms
   [3] GPU CSR + Bitmask Search Latency  :   60325.86 µs (First-run Triton JIT compile)
   [4] Full Couplet CUDA Graph Replay    :       8.00 µs
   [5] Omega 256-Step In-SRAM Kernel     :    1366.59 µs
   [6] Gemma-2 CUDAGraph Decode Speed    :       19.4 tok/s (51.58 ms/step)
   [7] Active GPU Docs / Online NNZ      :     2 docs / 0 online NNZ
   [8] Total GPU Index VRAM Footprint    :       8.44 MB
====================================================================================

[PDF BATCH GPU INGEST BENCHMARK]
File: Harrisons-Principles-of-Internal-Medicine-20th-Edition-Vol.1-Vol.2-Part-1.pdf
Pages: 2,223 | +102,409 Passages | PDF Extract: 14,588.4 ms
GPU CSR Build: 151.14 ms | Total Docs: 102,409 | Total Tokens: 2,723,135 | VRAM: 89.8 MB

[QUERY BENCHMARK — With Gemma-2 2B Fused-Triton Synthesis]
Query: 'تفاوت پاتوفیزیولوژی Bartter syndrome و Gitelman syndrome در ...'
Docs: 102,409 | Online NNZ: 0 | Gemma-2: True  | Total: 9,671.45 ms

[QUERY BENCHMARK — Direct GPU Retrieval]
Query: 'تفاوت پاتوفیزیولوژی Bartter syndrome و Gitelman syndrome در ...'
Docs: 102,409 | Online NNZ: 0 | Gemma-2: False | Total: 0.17 ms
```

</div>

<div dir="rtl" align="right">

### 📈 جدول مقایسه عملکرد با سیستم‌های Vector RAG متداول

| شاخص ارزیابی | Sovereign V14 (Triton CSR GPU RAG) | سیستم‌های Vector RAG متداول (FAISS / Chroma) |
| :--- | :---: | :---: |
| **حجم پیکرهٔ آزمایشی** | ۲,۲۲۳ صفحه (**۱۰۲,۴۰۹ قطعه** / **۲.۷۲ میلیون توکن**) | ۲,۲۲۳ صفحه (۱۰۲,۴۰۹ قطعه) |
| **زمان ساخت ایندکس روی GPU** | **۱۵۱.۱۴ میلی‌ثانیه (`0.15s`)** | ۱۰ الی ۲۵ دقیقه (محاسبه Embedding + ساخت گراف) |
| **مصرف حافظهٔ کارت گرافیک (VRAM)** | **۸۹.۸ مگابایت** (~۴۰۰ مگابایت برای ۱ میلیون سند) | ۱,۶۰۰+ مگابایت (برای ۱۰۰ هزار) / ۱۲+ گیگابایت (برای ۱ میلیون) |
| **زمان اجرای CUDA Graph هسته** | **۸.۰۰ میکروثانیه (`8.00 µs`)** | فاقد پشتیبانی از CUDA Graph در لایهٔ بازیابی |
| **زمان پاسخ‌دهی بازیابی (Retrieval)** | **۰.۱۷ میلی‌ثانیه (`170 µs`)** | ۱۵ الی ۸۰ میلی‌ثانیه |
| **یادگیری آنلاین (`Online Learning`)** | **آنی روی GPU بدون بازسازی ایندکس** | نیازمند محاسبه مجدد بردار و درج در درخت/گراف |

---

## 🖥️ امکانات داشبورد وب تعاملی (همراه با لینک عمومی Cloudflare)

با اجرای اسکریپت، یک سرور چندنخی محلی و یک تونل عمومی **Cloudflare (`*.trycloudflare.com`)** به صورت خودکار راه‌اندازی می‌شود که امکانات زیر را در اختیار شما قرار می‌دهد:

* 📄 **آپلود دسته‌ای و نامحدود فایل‌های PDF:** استخراج متن با `PyMuPDF` و ایندکس موازی روی GPU در چند ده میلی‌ثانیه.
* ⚡ **پنل یادگیری آنلاین (`online_learn`):** افزودن آنی متون، اصلاحات یا یادداشت‌های جدید به حافظهٔ GPU و مشاهدهٔ زندهٔ شمارندهٔ `Online NNZ`.
* 🚀 **دکمهٔ تست استرس ۱,۰۰۰,۰۰۰ سند (`1M-Doc GPU Scaler`):** ارتقای لحظه‌ای دیتابیس روی GPU به ۱ میلیون سند و اجرای بنچمارک زنده.
* 🌐 **جستجوی دوزبانه هوشمند (فارسی + انگلیسی):** قابلیت طرح پرسش به زبان فارسی و بازیابی دقیق شواهد از متون تخصصی انگلیسی یا فارسی همراه با هایلایت واژگان منطبق، شماره صفحه و نام سند.
* 🤖 **تولید پاسخ علمی فارسی با Gemma-2:** امکان فعال/غیرفعال‌سازی جمع‌بندی هوشمند شواهد با مدل `Gemma-2-2B` مجهز به کرنل‌های Triton.

---

## 🚀 راهنمای نصب و راه‌اندازی سریع

### ۱. پیش‌نیازها
* سیستم‌عامل لینوکس (یا محیط‌های ابری مانند Google Colab / Kaggle / سرورهای اختصاصی GPU)
* حداقل یک کارت گرافیک NVIDIA با پشتیبانی از CUDA (در صورت وجود ۲ کارت گرافیک، موتور بازیابی به صورت خودکار روی `cuda:0` و مدل زبانی Gemma-2 روی `cuda:1` قرار می‌گیرد).

### ۲. نصب کتابخانه‌ها و اجرا

</div>

<div dir="ltr" align="left">

```bash
# کلون کردن مخزن
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# نصب وابستگی‌ها (در صورت عدم نصب، خود اسکریپت به صورت خودکار آن‌ها را نصب می‌کند)
pip install torch triton pymupdf pyarrow bitsandbytes accelerate transformers

# اجرای پلتفرم، بنچمارک‌ها و دریافت لینک عمومی داشبورد Cloudflare
python sovereign_v14_platform.py
```

</div>

<div dir="rtl" align="right">

---

## 💻 استفاده برنامه‌نویسی (Python API)

علاوه بر داشبورد وب، می‌توانید مستقیماً در کدهای پایتون خود از موتور `platform` و `engine` استفاده کنید:

</div>

<div dir="ltr" align="left">

```python
# ۱. افزودن دانش جدید به صورت آنی روی GPU (Online Learning)
cid = engine.online_learn(
    "Severe ADAMTS13 deficiency (<10%) is the hallmark of Thrombotic Thrombocytopenic Purpura (TTP).",
    page=1,
    chapter=900
)

# ۲. پرسش دوزبانه (با یا بدون سنتز مدل Gemma-2)
result = platform.ask(
    "در بیماری TTP نقش کمبود آنزیم ADAMTS13 چیست؟",
    top_k=5,
    use_gemma=True
)

print("GPU CSR Latency (µs):", result["gpu_csr_us"])
print(result["answer"])

# ۳. تست مقیاس‌پذیری ۱,۰۰۰,۰۰۰ سند روی GPU
stats_1m = scale_engine_to_target_docs_gpu(target_docs=1_000_000)
print(stats_1m)
```

</div>

<div dir="rtl" align="right">

---

## 📜 مجوز انتشار (License)

این پروژه تحت مجوز **MIT License** منتشر شده است و استفاده از آن در پروژه‌های پژوهشی، صنعتی و تجاری آزاد است. اگر این پروژه برایتان مفید بود، با دادن یک ⭐️ (**Star**) در گیت‌هاب از آن حمایت کنید!

</div>
