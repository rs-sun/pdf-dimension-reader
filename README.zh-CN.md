# pdf-dimension-reader

[English](README.md) | 简体中文

面向 Siemens NX（旧称 UG）导出的矢量工程图 PDF 的尺寸识别与人工复核研究项目。核心通过工程字体的字形结构、笔画拓扑和字符排布寻找尺寸候选，经人工确认后生成标注并导出 Excel。输入为 PDF，不直接解析 NX/UG 原生 CAD 文件。

研究路线、核心方法和失败教训见 [中文研究梗概](RESEARCH_SUMMARY.md)（[English](RESEARCH_SUMMARY.en.md)）。该文档重新概括方法与设计取舍，不附带原始实验记录或工程图证据。

英文补充文档：[当前架构](docs/ARCHITECTURE.md)、[运行配置与资产绑定](docs/CONFIGURATION.md)、[贡献说明](docs/CONTRIBUTING.md)。

## 当前能力

- 普通尺寸通过真实形状锚点、方向走位、受控字形读取和候选合同进入人工确认工作台。
- 严格矢量路径不调用 OCR，也不用来源图纸的箭头、尺寸线或引线证明文字含义。
- 保留自主结构识别代码、有限字形结构统计及字符步长规则；不附带 NX/UG 字体文件或完整字形轮廓模板。
- GD&T 候选链与多页正确率、漏检、误检和性能仍需验收。基准字段始终留空。
- 公开源码未包含模型及受控轮廓模板，严格运行准备检查预期返回 `503 not_ready`。不能把这份源码称为可立即投入生产的最终版。

## 源码范围

包含识别与复核源码、有限字符结构及步长表、前端库及对应许可、合成合同样例和回归测试。没有工程图、图纸裁片、测量报告、NX/UG 字体二进制、完整轮廓模板、训练数据、模型权重或原开发历史。

官方 PDF.js 库内嵌一个仅含句点字形的通用字体加载测试文件；pdf-lib 内嵌公开的标准字体度量和编码表。它们属于上游公开发行内容，不是 NX/UG 字体或工程图数据，许可和来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

结构表记录量化宽高、图元类别、端点数量及字符步进等有限特征，不包含完整笔画坐标序列。源码保留针对工程字体结构和排布规律的识别方法。

## 本地运行

回归使用 Node.js 22；后端使用 Python 3.10 或更新版本。下列命令在源码根目录执行。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
PDF_READER_EAGER_LOAD_MODELS=0 FLASK_HOST=127.0.0.1 python backend/app.py
```

打开 `http://127.0.0.1:5000/`，后端会同时提供前端页面与 API。识别需要合法取得的本地运行资产，以及显式配置的 `YOLO_VIEW_MODEL`、`YOLO_VIEW_MODEL_SHA256`、`R33_M1_TEMPLATE_LIBRARY_PATH`。缺失或不满足严格合同的资产必须拒绝运行，不能用 OCR、测试模型或未固定身份的资产替代。表格接口和页面复核的可用性不代表识别准确率已经验收。

这份源码包不提供运行资产构建和验收的一键流程，受控模板的身份绑定被有意清空。即使外部模板成功加载，绑定不匹配仍会报告 `directional_walk_template_identity_mismatch` 并拒绝识别。重新构建受控模板后，还需要审核结构表与模板的身份绑定、更新相应受控清单及校验摘要，并验证严格合同。仅设置环境变量不能补齐这些步骤；不要直接复用未获再分发许可的字体、图纸或训练资产。

以上安装命令说明开发入口；整套依赖尚未完成全平台干净环境安装验收。OCR 模块为兼容代码，存在于源码中不代表严格矢量路径会调用它。

## 表格兼容范围

`.xlsx` / `.xlsm` 使用 openpyxl 读取缓存值。`.xls` 使用 python-calamine 0.8.2：保留普通文本、数值、空行空列位置和布尔值的数值表示；遇到日期、时间或时长单元格会明确拒绝导入，需要转换格式或整理这些单元格后再导入。解码器不再把 Excel 错误码作为数值，读为空串时按缺失处理。含公式的工作簿须先在电子表格软件中重算并保存。

## 回归

```bash
PYTHONDONTWRITEBYTECODE=1 node --test frontend_tests/*.test.mjs
python -m pip install -r backend/requirements-test.txt
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider backend/tests
```

后端测试覆盖尺寸语法、设置合同、受控模板准备检查、诊断输出边界、最终消费审批、HTTP 静态文件边界及表格解码。包含的合同样例与测试使用虚构数据、标量模拟或内存空白 PDF；没有附带工程图样本、截图或原始测量记录。

## 许可证

源码采用 **AGPL-3.0-only**，协议全文见 [LICENSE](LICENSE)。第三方库保留各自许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。源码许可不涵盖用户自行取得或生成的图纸、字体、模型与模板资产。

该项目为独立研究工具，与 Siemens 无官方关联或认可。
