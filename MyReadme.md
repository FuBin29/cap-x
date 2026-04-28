# 使用记录

## 目录

- [CaP-X 目录与架构分析](README_ARCHITECTURE_ANALYSIS.md)
- [官方 README](README.md)
- [配置说明](docs/configuration.md)
- [新增环境](docs/adding-environments.md)
- [新增 API](docs/adding-apis.md)
- [开发说明](docs/development.md)
- [LIBERO-PRO 任务](docs/libero-tasks.md)
- [BEHAVIOR/R1Pro 任务](docs/behavior-tasks.md)
- [真实 Franka Bringup](docs/real-franka.md)
- [RL 训练](docs/rl-training.md)
- [技能库编译](scripts/skill_library_compilation/README.md)

## 与官方库同步

```bash
git fetch upstream # 将官方库的所有分支更新下载到本地，但不会改动你目前的代码。
git checkout main # 切换到你的主分支（如果你在其他分支上）
git merge upstream/main # 将官方库的main分支合并到你的main分支，如果有冲突需要手动解决。
```
