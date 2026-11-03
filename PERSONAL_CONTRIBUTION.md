# 个人贡献与项目来源

## 项目背景

本项目基于 [Souvik Majumder 的 Multi-Agent-Medical-Assistant](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant) 进行团队二次开发，结合对话、文献检索、网络搜索和影像相关代理组织任务流程。

README 中的系统功能与技术栈描述属于项目整体；个人负责范围在本文件单独说明。

## 本人负责部分

本人主要负责 **Agent 工作流搭建与路由集成、适配**。提交签名为 `mobai`、`mobai828`。

### 工作流与路由

- 基于现有 LangGraph 工作流衔接对话、RAG、网络搜索和脑部影像相关代理。
- 参与文本对话与图像分析入口的拆分、脑肿瘤/脑卒中任务选择及影像接口适配。
- 项目同时保留自动决策与手动选择 Agent 的路径；基础自动决策逻辑来自开源上游，手动选择的相关提交签名为 `Trae Bot`，在团队集成中保留其来源。

个人签名提交可对应：[87d4345](https://github.com/mobai828/ai-agent/commit/87d4345f912dbdc54dfc30e5377a52d3873fa7de)、[1093193](https://github.com/mobai828/ai-agent/commit/10931938d35ed9354237b9806809f553da9c7a54)、[188ad81](https://github.com/mobai828/ai-agent/commit/188ad811e9dbd84dc3a7f6e459f10bdf82d47043)。代码入口为 `agents/agent_decision.py` 和 `app.py`。

### 会话与状态管理

个人提交 [db6d07c](https://github.com/mobai828/ai-agent/commit/db6d07cf6121fe9642b8ab151da251472f46d00e) 中包含以下运行时代码改造，提交标题虽然以 docs 开头，实际差异也修改了业务代码：

- 将请求中的 `session_id` 传入 LangGraph，作为 checkpoint 的 `thread_id`。
- 优先使用 SQLite 保存会话状态，初始化失败或配置要求时回退到内存保存。
- 缓存编译后的图与 checkpointer，避免每次请求重新构建。
- 使用 `RemoveMessage` 将超出窗口的历史写回状态，并加入过期会话清理逻辑。

这些内容已通过静态代码和提交差异核对；会话隔离、重启恢复和清理行为仍需运行测试验证。

### 贡献范围

以上描述本人负责与参与的工程部分。基础多代理框架、原有 RAG 实现和其他作者的代码保留原始来源；影像模型训练、医学准确率及完整系统成果需分别依据团队分工与验证材料说明。

## 来源与版本

- 开源上游：[Souvik Majumder / Multi-Agent-Medical-Assistant](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant)。
- 团队版本来源记录：[Gitee medical_image](https://gitee.com/obelisktvt/medical_image)。
- 现有个人维护仓库：[mobai828/ai-agent](https://github.com/mobai828/ai-agent)。
- 关联的早期个人仓库：[mobai828/agentgithub](https://github.com/mobai828/agentgithub)。
- 本次个人整理以提交 `3ef5d5d8363dda6c50361f1ed8017e324093567a` 及已有的本地修改为基础。
- 项目保留原有 LICENSE 与来源信息；个人整理不改变已有代码的归属。

## 当前验证状态

2026-11-03 已完成离线启动、8 项离线回归测试，以及浏览器页面与本地聊天检查。模型推理、远端服务、RAG 数据、在线路由和 SQLite 重启恢复仍未验证。

个人整理阶段另修复了“离线模式仍要求在线密钥并加载在线组件”的启动问题，补充离线模式标识、端点限制和回归测试。这些维护内容与前述历史负责部分分列，见 [本地演示记录](docs/LOCAL_DEMO.md)。

后续演示将区分实际可运行功能、外部服务依赖和待完成内容；结果以记录的验证环境与证据为准。
