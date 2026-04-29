# 企业知识库问答 + 自动办事 Agent

这是一个可运行 MVP：

- 上传 PDF / DOCX / TXT / MD
- 自动切块
- 调用 OpenAI Embeddings 生成向量
- 存入本地 SQLite
- 支持知识库问答
- 支持自动生成待办清单、邮件草稿、会议纪要、项目推进表、风险清单、执行步骤
- 输出结果保存为 Markdown

## 1. 安装 Python

建议 Python 3.10 或更高版本。

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 OpenAI API Key：

```bash
cp .env.example .env
```

Windows 用户可以直接复制文件并改名为 `.env`。

## 4. 启动

```bash
streamlit run app.py
```

启动后浏览器会自动打开界面。

## 5. 使用流程

1. 在「上传入库」里上传企业资料。
2. 等待入库完成。
3. 到「知识库问答」提问。
4. 到「自动办事」生成邮件草稿、待办清单等结果。

## 注意

- 扫描版 PDF 需要 OCR，本 MVP 暂不处理图片文字。
- 本项目是演示版，不含企业权限系统、用户隔离、审计日志、加密存储。
- 企业真实部署时，不建议直接把敏感合同、客户隐私、商业机密发到外部模型接口，需要先做合规评估。
