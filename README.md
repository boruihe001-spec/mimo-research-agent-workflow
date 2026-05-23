# MiMo Research Agent Workflow

本项目是一个面向科研学习、工程实践与学术材料生产的轻量级 AI Agent 工作流原型，计划接入 Xiaomi MiMo API，用于测试国产大模型在中文科研写作、代码生成、长文档理解和 Agent 工具调用场景中的表现。

## Project Background

在机器学习、深度学习和毕业论文相关项目中，常见任务包括代码生成与调试、实验日志整理、论文结构检查、PPT 内容生成、答辩问题模拟和项目文档问答。这些任务通常具有上下文长、多轮迭代频繁、代码与文档交叉修改等特点，适合作为 AI Agent 的真实落地场景。

## Planned Features

- 科研代码辅助：Python、机器学习、深度学习代码生成、调试与解释
- 文档处理 Agent：论文、课程报告、表格材料的结构检查与内容润色
- PPT 辅助 Agent：根据论文内容生成答辩 PPT 大纲、讲稿和备份页
- 知识库问答：基于论文、实验记录和项目文档进行问答与复盘
- 运行日志记录：保存任务输入、模型输出和人工校验记录

## Current Status

当前项目处于原型设计与工作流验证阶段。已完成初步目录结构、Prompt 模板、示例输入输出和终端运行流程。后续计划接入 Xiaomi MiMo API，进一步测试其在真实科研与工程学习场景中的可用性。

## Why Xiaomi MiMo API

希望测试 Xiaomi MiMo API 在以下任务中的表现：

- 中文科研写作与润色
- 长上下文文档理解
- Python / ML / DL 代码生成与调试
- Agent 编程工具接入
- 多轮任务规划与结果复盘

## Demo Tasks

### 1. Thesis Review Agent

输入论文摘要或章节内容，输出结构优化建议、表达修改建议和答辩风险点。

### 2. Code Debug Agent

输入 Python 报错信息和代码片段，输出错误原因、修改建议和修复后的代码。

### 3. PPT Outline Agent

输入论文主题和章节结构，输出答辩 PPT 页面规划、讲稿要点和可能问题。

## Future Plan

- 接入 Xiaomi MiMo API
- 增加终端调用示例
- 增加多轮 Agent 工作流
- 增加项目文档知识库问答
- 整理 MiMo 与其他模型在科研任务中的对比记录
