# 架构合规整改计划

审计日期: 2026-08-02
关联 Issue: #1
违规等级: P1
合规评级: C+

层级定位: 一、底层算力基座 - 统一模型仓库与智能调度中心
核心职责: 模型元数据管理、模型下载、模型版本管理

违规项与整改:
1. inference router 直接代理推理请求 - 移除, 调用方直接访问fusion-mlx或经fusion-gateway路由 - P1-S1
2. cluster router 集群节点管理 - 移除至fusion-multi-node - P1-S1
3. evaluations/benchmarks router 评测结果 - 移除至fusion-bench - P1-S2
4. quantize执行逻辑 - 量化执行委托fusion-mlx, 只保留配置元数据 - P1-S2
5. deployments灰度发布 - 考虑迁至fusion-gateway - P1-S3

合规标准: model-hub应只包含模型元数据CRUD/模型下载/版本管理/格式转换配置, 不应代理推理/管理集群/执行评测/灰度发布
