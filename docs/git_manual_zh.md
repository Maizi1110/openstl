# Git 操作手册（含用户名邮箱配置）

## 1. 你要记住的主线

1. 家里电脑：`add -> commit -> push`（把代码传到 GitHub）
2. 学校电脑：`pull`（把 GitHub 最新代码拉下来）

## 2. 第一次使用 Git（先做这个）

### 2.1 配置用户名和邮箱（全局）

```bash
git config --global user.name "Maizi1110"
git config --global user.email "你的GitHub邮箱"
```

作用：这两项会写进提交记录里，别人能看到是谁提交的。

### 2.2 检查是否配置成功

```bash
git config --global --get user.name
git config --global --get user.email
```

### 2.3 如果填错了，直接覆盖

```bash
git config --global user.name "新名字"
git config --global user.email "新邮箱"
```

### 2.4 只想当前仓库用不同身份（可选）

先进入仓库目录，再执行：

```bash
git config user.name "当前仓库名字"
git config user.email "当前仓库邮箱"
```

说明：不加 `--global` 就是仅当前仓库生效。

## 3. 常用指令说明（逐条）

1. `git status`  
作用：看当前有哪些文件改了、哪些还没提交。

2. `git add .`  
作用：把当前目录下所有改动放进“暂存区”。

3. `git commit -m "说明文字"`  
作用：把暂存区内容打成一次提交（快照）。

4. `git push origin master`  
作用：把本地提交上传到 GitHub 的 `master` 分支。

5. `git pull origin master`  
作用：把 GitHub 上 `master` 的最新代码拉到本地。

6. `git clone 仓库地址`  
作用：第一次在新电脑下载整个仓库。

7. `git branch --show-current`  
作用：看当前所在分支。

8. `git remote -v`  
作用：看当前仓库绑定的远程地址。

9. `git remote set-url origin 仓库地址`  
作用：远程地址绑错时，改成正确地址。

## 4. 你的固定流程（最简版）

### 4.1 家里电脑改完后

```bash
git status
git add .
git commit -m "本次修改说明"
git push origin master
```

### 4.2 学校电脑开工前

```bash
git pull origin master
```

## 5. 学校电脑第一次拉代码（只做一次）

```bash
git clone https://github.com/Maizi1110/openstl.git
cd openstl
git checkout master
```

可额外检查远程地址是否正确：

```bash
git remote -v
```

## 6. 常见问题处理

### 6.1 `nothing to commit`
意思：没有新改动，不需要 commit。

### 6.2 `failed to push`（远程比你新）

```bash
git pull --rebase origin master
git push origin master
```

### 6.3 `Authentication failed`
意思：认证失败（密码方式不可用或凭据失效）。

处理：
1. 使用 GitHub Token 作为密码。  
2. 或清理旧凭据后重新登录。

### 6.4 本地仓库绑错远程

```bash
git remote set-url origin https://github.com/Maizi1110/openstl.git
git remote -v
```

## 7. 约定

- 主分支使用 `master`。
- 仓库地址固定为 `https://github.com/Maizi1110/openstl.git`。