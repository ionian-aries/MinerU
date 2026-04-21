# MinerU Custom 索引

本目录仅保留统一入口文件：

- `__init__.py`：对外导出 custom 能力
- `registry.py`：统一注册与路由入口（`resolve_*`）
- `README.md`：功能索引

## 当前定制功能入口

- [`storage/`](./storage/)
  - 文档：[`storage/README.md`](./storage/README.md)
  - 代码：[`storage/storage.py`](./storage/storage.py)

- [`discard_policy/`](./discard_policy/)
  - 文档：[`discard_policy/README.md`](./discard_policy/README.md)
  - 代码：[`discard_policy/discard_policy.py`](./discard_policy/discard_policy.py)

- [`enhance_mvp/`](./enhance_mvp/)
  - 代码：`enhance_mvp/pipeline.py` + `enhance_mvp/stages_*.py`
  - 能力：文档语义增强 MVP（section/doc 摘要与关键词）
  - 输出：`*_enhance.json` + 最终增强版 `*.md`
  - 注意：增强模式下必须可调用 LLM，未配置 provider 将直接报错（无 heuristic/mock 回退）

## 说明

- 业务代码统一通过 `custom/registry.py` 使用定制能力。
- 各功能的作用、效果、配置与示例请查看对应子目录 README。
 - 二开配置入口统一为 `--custom-config` 指定的单一配置文件（详见 `notes/18`）。不再使用环境变量或 `~/mineru.json` 承载二开配置。
