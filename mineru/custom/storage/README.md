# Storage Customization

## 作用

为 MinerU 提供双通道输出存储能力：

- 图片通道（image）
- 文档通道（doc：md/json/pdf等）

支持本地与 S3 兼容对象存储（如阿里 OSS）。

## 效果

- 可以分别配置图片与文档的存储后端。
- 支持通过 **单一二开配置文件**（见 `notes/18`）方式接入。

## 如何使用

- CLI：`mineru/razel-mineru parse --custom-config /path/to/custom.yaml`
- FastAPI：启动 `mineru-api --custom-config /path/to/custom.yaml`
- 说明：storage 行为只受 `custom.yaml` 中 `storage` 段控制，不支持请求级/环境变量覆盖。

## 示例

```yaml
storage:
  image:
    backend: s3
    s3_bucket: your-bucket
    s3_prefix: mineru/job-001/images
    s3_ak: "..."
    s3_sk: "..."
    s3_endpoint_url: https://oss-cn-beijing.aliyuncs.com
    s3_addressing_style: virtual
    ref_prefix: https://your-bucket.oss-cn-beijing.aliyuncs.com
  doc:
    backend: local
```
