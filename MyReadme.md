# 使用记录

## 与官方库同步

```bash
git fetch upstream # 将官方库的所有分支更新下载到本地，但不会改动你目前的代码。
git checkout main # 切换到你的主分支（如果你在其他分支上）
git merge upstream/main # 将官方库的main分支合并到你的main分支，如果有冲突需要手动解决。
```