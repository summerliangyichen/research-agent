# News Agent 知识库路线大纲

## 当前起点

当前 MVP 已经具备最小闭环：

```text
用户 query
  -> Tavily 搜索相关 URL
  -> LangGraph 调用工具
  -> crawl_webpage 读取网页
  -> LLM 输出总结
```

下一阶段目标是把它从“搜索总结 demo”升级成“新闻知识库生成器”：

```text
自动获取新闻
  -> 生成 Obsidian 风格 Markdown 笔记
  -> 建立新闻、实体、主题、故事线之间的双链关系
  -> 后续接入 embedding / RAG 做语义检索和知识库问答
```

## 总体目标

做一个个人新闻知识库 agent：

```text
输入主题 / 定时运行
  -> 搜索新闻
  -> 抓取或读取正文
  -> 提取结构化信息
  -> 生成 Markdown 新闻笔记
  -> 自动关联人物、公司、主题、事件线
  -> 在 Obsidian 中形成可浏览、可检索、可追溯的新闻知识图谱
```

## 目标架构

```text
Tavily / RSS / 手动 URL
        |
        v
News Discovery
        |
        v
Content Reader
        |
        v
Article Extractor
        |
        v
Relation Builder
        |
        v
Obsidian Markdown Writer
        |
        v
SQLite / JSONL Index
        |
        v
Embedding Search
        |
        v
RAG QA
```

## 目录规划

```text
news-agent/
  main.py
  graph.py
  tools.py
  store.py
  schemas.py
  prompts.py
  scheduler.py
  README.md

news-vault/
  articles/
  entities/
  topics/
  stories/
  daily/
```

## 阶段 1：稳定新闻简报

目标：运行一次，稳定生成一份带来源链接的新闻结果。

流程：

```text
用户输入 query
  -> Tavily 搜索相关新闻
  -> 获取 URL / title / snippet
  -> 读取网页正文或 Tavily 摘要
  -> LLM 生成中文总结
  -> 保存结果
```

产物：

```text
outputs/latest.md
outputs/runs.jsonl
```

最低验收标准：

```text
运行 main.py，输入一个主题，能生成一份包含标题、摘要、来源 URL 的 Markdown 简报。
```

## 阶段 2：Obsidian 新闻笔记

目标：每篇新闻保存成一条 Obsidian 友好的 Markdown 笔记。

单篇新闻笔记格式：

```markdown
---
title: "新闻标题"
source: BBC
url: https://example.com/news/article
published_at: 2026-05-14
fetched_at: 2026-05-14
tags:
  - news/technology
entities:
  - Meta
  - Apple
topics:
  - Privacy
  - Smart Glasses
story: Meta 智能眼镜隐私争议
---

# 新闻标题

## 摘要

这里写中文摘要。

## 关键事实

- 事实 1
- 事实 2
- 事实 3

## 相关实体

- [[Meta]]
- [[Apple]]
- [[Privacy]]

## 关联新闻

- [[另一篇相关新闻]]

## 原文

[原文链接](https://example.com/news/article)
```

最低验收标准：

```text
运行 main.py 后，自动在 news-vault/articles/ 里生成一篇带 YAML、摘要、来源和 [[双链]] 的新闻笔记。
```

## 阶段 3：关系构建

目标：让新闻之间自动关联起来。

第一版先用规则关联，不急着上 embedding：

```text
同一实体：都提到 Meta / OpenAI / Trump
同一主题：都属于 AI / 隐私 / 经济
同一故事线：都是某个事件的连续报道
时间接近：几天内连续出现
URL 重复：同一新闻去重
标题相似：疑似重复报道
```

生成的关系：

```text
[[实体]]
[[主题]]
[[故事线]]
[[相关新闻]]
```

最低验收标准：

```text
新入库新闻能自动写入 entities/topics/story，并在笔记中生成 Obsidian 双链。
```

## 阶段 4：日报与索引

目标：每天自动生成一个入口笔记。

示例：

```text
news-vault/daily/2026-05-14.md
```

内容格式：

```markdown
# 2026-05-14 新闻日报

## AI

- [[2026-05-14-openai-model-update]]

## 隐私

- [[2026-05-14-meta-smart-glasses-privacy]]

## 国际

- [[2026-05-14-global-politics-example]]
```

最低验收标准：

```text
每次入库新闻后，能自动更新当天 daily note。
```

## 阶段 5：结构化索引

目标：方便查询、去重、后续 embedding 和 RAG。

先保留 Markdown 作为主数据，JSONL 作为运行日志：

```text
articles/*.md
outputs/runs.jsonl
```

稳定后再加 SQLite：

```text
articles
entities
topics
stories
article_entities
article_topics
article_relations
```

最低验收标准：

```text
能根据 URL 去重，避免同一篇新闻重复生成笔记。
```

## 阶段 6：Embedding 相似检索

目标：发现语义相关但关键词不完全一致的新闻。

不要把 embedding 当成第一版核心存储。Markdown/SQLite 仍然是主数据，embedding 只是“语义索引层”，用于发现隐含关系。

### 什么时候上 embedding

建议等知识库至少积累 20-50 篇新闻后再做。太早上 embedding 会有两个问题：

```text
文章结构还不稳定 -> 向量质量不稳定
历史新闻太少 -> 相似检索没有明显价值
```

### embedding 的输入文本

每篇新闻不要直接拿完整网页正文做 embedding。第一版建议使用“浓缩后的语义文本”：

```text
title + summary + key_facts + entities + topics + story
```

推荐格式：

```text
Title: Smart glasses are an invasion of privacy
Source: BBC
Published: 2026-05-13
Story: Meta 智能眼镜隐私争议
Topics: Privacy, Smart Glasses, AI Hardware
Entities: Meta, Ray-Ban, Apple, Google, Snap
Summary: ...
Key facts:
- ...
- ...
```

这样做的好处：

```text
噪音少：去掉导航、广告、推荐链接
关系强：实体、主题、故事线会显著影响相似度
成本低：短文本 embedding 更便宜
更稳定：不同新闻站点正文格式差异不会污染向量
```

### 是否需要 chunk

第一版不建议复杂 chunk。每篇新闻先生成一个 article-level embedding：

```text
article_embedding = embedding(title + summary + key_facts + entities + topics   y)
```

后面如果要支持深度 RAG，再加 chunk-level embedding：

```text
article-level embedding：用于找相关新闻、去重、故事线合并
chunk-level embedding：用于回答细节问题时检索具体段落
```

chunk 设计可以晚点做：

```text
chunk_size: 500-800 中文字 / 300-600 英文词
overlap: 80-150 字
chunk_source: summary + excerpt，不保存完整原文时只切正文节选
```

### 向量存储设计

第一版可以先用 SQLite 保存元数据，向量用本地文件或 SQLite blob。后续再换专门向量库。

最小结构：

```text
article_embeddings
  id
  article_id
  model
  embedding_text_hash
  embedding
  created_at
```

如果后续有 chunk：

```text
article_chunks
  id
  article_id
  chunk_index
  text
  token_count

chunk_embeddings
  id
  chunk_id
  model
  embedding
  created_at
```

### 相似新闻关联流程

新新闻入库时：

```text
1. 生成 Article 结构化数据
2. 生成 embedding_text
3. 调用 embedding 模型得到向量
4. 在历史 article_embeddings 中搜索 top_k
5. 过滤 URL 完全相同的结果
6. 结合规则分数重新排序
7. 超过阈值则写入 article_relations
8. 在 Markdown 的 “关联新闻” 区域写入 [[wikilink]]
```

推荐第一版打分：

```text
final_score =
  0.70 * embedding_similarity
  + 0.15 * shared_entity_score
  + 0.10 * shared_topic_score
  + 0.05 * time_proximity_score
```

阈值建议先保守：

```text
similarity >= 0.82：可自动关联
0.72 <= similarity < 0.82：候选关联，先记录不自动写双链
similarity < 0.72：忽略
```

这些阈值不是固定真理，后面要根据实际新闻库微调。

### embedding 适合的用途

```text
相似新闻推荐
语义去重
事件线合并
自然语言搜索
相关新闻自动链接
```

具体例子：

```text
标题 A：Meta smart glasses face privacy backlash
标题 B：Women say they were secretly filmed by AI glasses

关键词不完全一样，但 embedding 能识别它们都属于：
  -> 智能眼镜
  -> 隐私争议
  -> Meta / Ray-Ban
  -> 偷拍与公共空间
```

### embedding 不应该做的事

```text
不要用 embedding 替代 URL 去重
不要只靠 embedding 判断事实真假
不要把完整网页噪音直接丢进向量
不要让 LLM 看到相似新闻就强行建立关系
不要没有来源链接就做 RAG 回答
```

### 最低验收标准

```text
新增一篇新闻时，能找到 3-5 篇语义相近的历史新闻，并能把高置信关联写入 Markdown 的 “关联新闻” 区域。
```

## 阶段 7：RAG 问答

目标：基于本地新闻知识库回答问题，并附来源。

RAG 不是“让模型记住新闻”，而是每次回答前从本地知识库里找证据，再让模型基于证据回答。

### RAG 适合解决的问题

```text
最近一周 AI 新闻有什么变化？
Meta 智能眼镜隐私争议有哪些进展？
OpenAI 最近产品路线有什么变化？
某个事件从哪篇新闻开始？
这个新闻和之前哪些新闻有关？
同一事件不同媒体的叙述有什么差异？
```

### RAG 不适合解决的问题

```text
实时新闻：如果知识库今天还没采集，RAG 不会凭空知道
无来源判断：没有证据的内容不能回答成事实
复杂事实核查：只能基于已入库材料，不能替代完整调查
需要登录/付费墙全文的问题：除非已合法采集到可用摘要或节选
```

### 第一版 RAG 流程

```text
用户问题
  -> 生成 query embedding
  -> 检索 top_k article embeddings
  -> 用关键词/实体/时间过滤结果
  -> 取回 Markdown 笔记中的摘要、关键事实、正文节选、来源
  -> 拼接上下文
  -> LLM 只基于上下文回答
  -> 输出答案 + 引用来源 + 相关笔记链接
```

### 检索策略

第一版不要只做纯向量检索，建议 hybrid retrieval：

```text
向量检索：找语义相近文章
关键词检索：匹配实体、主题、标题、来源
时间过滤：限定最近 7 天 / 30 天 / 某个日期范围
来源过滤：只看 BBC / Reuters / 官方博客等
```

推荐顺序：

```text
1. query embedding 找 top 20
2. 规则过滤到 8-12 篇
3. 可选 LLM rerank 到 5-8 篇
4. 拼上下文回答
```

### 上下文格式

传给 LLM 的上下文要短、干净、可引用：

```text
[1]
title: ...
source: BBC
published_at: 2026-05-13
url: https://...
note: [[2026-05-13-meta-smart-glasses-privacy]]
summary: ...
key_facts:
- ...
- ...
excerpt: ...

[2]
...
```

回答时要求：

```text
只基于提供的 context 回答
不确定就说明知识库中没有足够证据
每个关键结论附来源编号
不要编造没有出现在 context 里的细节
```

### 输出格式

普通问答：

```markdown
## 回答

...

## 依据

- [1] BBC, 2026-05-13, [[note-name]]
- [2] Reuters, 2026-05-14, [[note-name]]

## 相关笔记

- [[Meta]]
- [[Privacy]]
- [[Meta 智能眼镜隐私争议]]
```

事件追踪类问题：

```markdown
## 时间线

- 2026-05-10：...
- 2026-05-13：...
- 2026-05-14：...

## 主要变化

...

## 仍不确定的地方

...
```

### RAG 的最小代码模块

```text
embeddings.py
  build_embedding_text(article)
  embed_text(text)
  search_similar_articles(query, top_k)

retriever.py
  retrieve_context(query, filters)
  rerank_articles(query, candidates)

rag.py
  answer_with_sources(query)
  build_context(articles)

store.py
  read_article_note(article_id)
  read_article_index()
```

### 什么时候使用 article-level，什么时候使用 chunk-level

```text
article-level：
  适合“找相关新闻”“这个主题最近有哪些进展”“这篇新闻和谁相关”

chunk-level：
  适合“某篇报道里具体说了什么”“有哪些证据支持这个结论”“对比多个报道细节”
```

第一版先做 article-level RAG。等你发现回答细节不够准，再加 chunk-level。

### 最低验收标准

```text
能从本地新闻笔记中检索 5-8 篇相关文章，生成带来源编号、原文链接和 Obsidian 笔记链接的回答。
```

## 阶段 8：自动化运行

目标：固定主题每天自动更新。

流程：

```text
固定主题列表
  -> 每天定时运行
  -> 搜索相关新闻
  -> 跳过重复新闻
  -> 生成文章笔记
  -> 更新 daily note
  -> 记录错误日志
```

主题配置示例：

```python
TOPICS = [
    "AI latest news",
    "OpenAI news",
    "technology privacy news",
]
```

最低验收标准：

```text
每天自动生成一份 daily note，并把新新闻入库。
```

## 推荐开发顺序

```text
1. 保存单篇 Markdown 新闻笔记
2. 生成 entities / topics / story 双链
3. 生成 daily note
4. 加 JSONL 运行日志
5. 加 URL 去重
6. 加 SQLite 索引
7. 积累 20-50 篇新闻
8. 上 embedding 相似检索
9. 上 RAG 问答
10. 加定时任务和失败重试
```

## 当前下一步

优先实现：

```text
运行 main.py，输入一个主题，自动在 news-vault/articles/ 里生成一篇 Obsidian Markdown 新闻笔记。
```

建议新增模块：

```text
schemas.py    定义 Article 等结构
store.py      负责保存 Markdown / JSONL
prompts.py    负责结构化抽取 prompt
```

## 待确认问题

1. Obsidian vault 放在哪里？

   已确认：

   ```text
   D:\Summer\代码库\news-agent\news-vault
   ```

2. 每次运行处理几篇新闻？

   已确认：

   ```text
   5-10 篇
   ```

3. 知识库主要语言是什么？

   已确认：

   ```text
   中文摘要 + 英文原文标题和来源保留
   ```

4. 新闻来源范围要不要限制？

   可选：

   ```text
   只要 Tavily 搜索结果
   固定白名单媒体
   Tavily + RSS + 手动 URL
   ```

5. 主题是用户每次输入，还是固定订阅列表？

   可选：

   ```text
   每次手动输入
   固定主题列表
   两者都支持
   ```

6. 关系链接先用规则，还是马上接 embedding？

   默认建议：

   ```text
   先规则双链，积累 20-50 篇后再上 embedding
   ```

7. 是否需要保留原文全文？

   已确认：

   ```text
   保存摘要和正文节选，不保存完整正文
   ```

8. 去重标准以什么为准？

   默认建议：

   ```text
   URL 精确去重 + 标题相似度辅助判断
   ```
