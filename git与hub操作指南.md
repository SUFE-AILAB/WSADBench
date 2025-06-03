下面是 Git 协作开发中的完整流程命令，涵盖 **创建分支、修改、推送、提交 PR** 的标准操作。以下以 GitHub 为例，也适用于 Gitee 等平台。

---

## 🧱 前提准备

先确保你已经 `clone` 了项目：

```bash
git clone https://github.com/your-org/project-name.git
cd project-name
```

如果本地已经有了项目文件夹，可以通过 `git status` 确定是否已经位于一个有效的git仓库内。

---

## 🔀 步骤 1：创建并切换到自己的分支

```bash
git checkout -b your-branch-name
```

示例：

```bash
git checkout -b feature/add-login-api
```

---

## ✏️ 步骤 2：开发和修改代码

使用你熟悉的编辑器修改代码，然后保存。

---

## ✅ 步骤 3：查看修改并提交

```bash
git status                         # 查看修改
git add .                          # 添加所有修改（或指定文件）
git commit -m "feat: 添加登录接口"
```

---

## 📤 步骤 4：推送分支到远程仓库

```bash
git push origin your-branch-name
```

示例：

```bash
git push origin feature/add-login-api
```

---

## 🔁 步骤 5：在 GitHub 上提交 Pull Request（PR）

推送成功后，去 GitHub 页面：

* 会提示你 “Compare & pull request”
* 填写 PR 标题、描述
* 选择目标分支（如 `main` 或 `develop`）
* 提交 PR

---

## 📌（可选）步骤 6：同步主分支更新（防冲突）

在等待 PR 合并时，若主分支有更新，建议定期同步：

```bash
git checkout main
git pull origin main
git checkout your-branch-name
git merge main           # 或者 git rebase main
```

注意，必须要先 checkout 到 main 分支，然后再 pull . 如果直接再当前分支下 git pull origin main， 这等价于 

```bash
git fetch origin main
git merge FETCH_HEAD
```

这是错误操作！它会把远程 main 的代码合并进当前分支，不是“更新 main 分支”，而是“用 main 更新你当前的分支”。
我们推荐更新本地的main分支，再merge或rebase操作解决冲突。

---

## 🧹（可选）步骤 7：PR 合并后清理分支

```bash
git branch -d your-branch-name              # 删除本地分支
git push origin --delete your-branch-name  # 删除远程分支（如需要）
```


## 常用概念


### ✅ 分支（Branch）

分支是代码的并行开发线。你可以在一个分支上开发新功能，而不影响主线（如 `main`）。多个分支可以并行存在，最终通过合并合并到主分支。

---

### ✅ 本地仓库（Local Repository）vs 远端仓库（Remote Repository）

**本地仓库** 是你电脑上的 Git 仓库，包含项目代码和 Git 提交历史。你在本地进行代码修改、提交、分支操作等。
**远端仓库** 是托管在 GitHub、Gitee 等服务器上的共享仓库，用于多人协作。你的本地仓库可以与远端仓库同步代码。

---


### ✅ origin

`origin` 是远端仓库的默认名称，是本地对远程仓库地址的简写别名。例如，`git push origin main` 表示将 `main` 分支推送到远端的 `origin` 仓库。

---

### ✅ Pull Request（简称 PR）

PR 是向远端仓库提交代码合并请求的操作。通常是你在新分支上开发功能后，向主分支发起 PR，请求合并代码。适用于团队协作代码审查。

---

### ✅ rebase vs merge

`rebase` 和 `merge` 都用于将一个分支的更改整合到另一个分支，但方式不同：

* `merge` 会保留两个分支的提交历史，通过一个“合并节点”把它们连接起来，历史结构是分叉的；
* `rebase` 会把当前分支的提交“平移”到目标分支之后，历史变成一条直线，看起来像是一次连续开发。

**区别**在于：`merge` 保留并行开发的真实历史，适合团队协作审计；而 `rebase` 保持提交历史整洁线性，适合个人开发或提交前整理历史。

简言之：**merge 保留分支痕迹，rebase 让历史看起来像没有分支过。**

---

### ✅ pull

`pull` 等价于 `fetch + merge`，即从远程仓库拉取最新代码并合并到当前分支。常用于同步远程更新。

---

### ✅ fetch

`fetch` 只拉取远程仓库的更新到本地，不合并。适用于先查看变更，再决定是否合并或 rebase。

---

