# Discard Policy Customization

## 作用

定义解析结果中哪些 BlockType 需要丢弃（如页眉、页脚、页码等）。

## 效果

- 控制最终 markdown/content_list/middle_json 的保留内容范围。
- 统一作用于 pipeline / vlm / hybrid 后端中的丢弃判断逻辑。

## 如何使用

- 通过 **单一二开配置文件**（见 `notes/18`）控制，在其中配置 `discard.types`。
- CLI：`mineru/razel-mineru parse --custom-config /path/to/custom.yaml`
- FastAPI：启动 `mineru-api --custom-config /path/to/custom.yaml`
- 说明：discard 行为只受 `custom.yaml` 中 `discard` 段控制，不支持请求级/环境变量覆盖。

## 示例

```bash
mineru \
  -p "/path/to/input.pdf" \
  -o "/path/to/output" \
  --backend pipeline \
  --custom-config "/path/to/custom.yaml"
```

```yaml
discard:
  types:
    - header
    - footer
    - page_number
```
