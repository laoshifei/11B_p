# 11B_p

使用 Geant4、ROOT 和 C++17 模拟质子入射硼靶的核反应、三 α 衰变和探测器响应。当前工程位于 `g4/`，由原 `xc_260921/` 整理而来；历史日期版本通过 Git 提交和标签保留。

## 目录

| 路径 | 内容 |
| --- | --- |
| `g4/HB.cc`、`g4/CMakeLists.txt` | 程序入口、线程与随机种子、构建配置 |
| `g4/include/`、`g4/src/` | 反应模型、衰变运动学、探测器与 ROOT 数据结构 |
| `g4/macros/`、`g4/batch.sh` | 运行配置、批量运行、线程 ROOT 文件合并 |
| `g4/analysis/` | α 能谱、通道、Dalitz 图、探测器响应、效率与拟合分析 |
| `g4/cs_model/` | 截面模型计算与绘图脚本 |
| [HISTORY.md](HISTORY.md) | 来源目录、日期标签、快照提交、版本差异与回溯方法 |

`jupyter/`、`papers/`、`presentation/`、所有 `.ipynb`、ROOT 数据、构建目录、生成图像、日志、导出表格及压缩包只在本地保存，排除规则见 [.gitignore](.gitignore)。克隆仓库得到源码、宏、分析脚本和说明文件，运行后会在本地产生数据和图像。

## 当前模型与输出

- `H11BCrossSection` 和 `H11BReaction` 实现 162/675 keV 共振分量、顺序三 α 衰变、直接三 α 分支及 γ 捕获；各通道是否启用由宏决定。
- 675 keV 分量采用含能量相关库仑穿透率的 Breit-Wigner 截面；三 α 衰变支持 `symmetrizedCoherentL1L3`，由 `/h11b/` 命令配置。
- 靶、束斑、Si 阵列和虚拟球响应可在宏中配置，另包含 LaBr3、HPGe 响应。几何相关设置应放在 `/run/initialize` 之前。
- ROOT 保存反应、运行信息和按配置启用的探测器响应；字段以 `g4/include/DataStructure.hh`、`g4/src/RootIO.cc` 为准。分析旧数据时核对对应版本及 ROOT 内的运行配置。
- `/run/beamOn` 表示入射事件数，实际反应数由 `reaction` 树统计。偏置采样下的原始计数不直接等于绝对物理产额。

162 keV 的宽 `8Be(2+)` 分支新增可选的纯 `L=2` Breit–Wigner 振幅控制：

```text
/h11b/162AlphaDecayModel boseSymmetricL2BW
/h11b/162BoseRadialBins 1024
/h11b/162BoseMaxAttempts 100000
```

`singlePathL2BW`、`incoherentL2BW`、`boseSymmetricL2BW` 分别采用单路径、三配对强度相加、三配对复振幅相加后平方。三者使用相同线形和角动量耦合；默认仍为 `legacyLegendreA2A4`。新模型假定未极化母态，不使用旧次级 A2/A4 参数，并拒绝非零的 alpha1 初级角分布系数。仅替换 162 的 alpha1 分支；基态、直接过程、γ 和 675 路径保持原有处理。

相干采样从非相干路径混合分布提出事件，以 `I_coh/(3 I_inc)` 接受，解析上界为 1；越界或超出尝试上限时报错。采用现有穿透率及 BW 线形、BW 复相位，尚未包含 R-matrix 能移、外部库仑／硬球相位及三体库仑修正，不能称为完整 Laursen M4 或已获实验验证的模型。方法依据：K. L. Laursen et al., *Complete kinematical study of the 3α breakup of the 16.11 MeV state in 12C*, EPJ A **52**, 271 (2016), DOI: [10.1140/epja/i2016-16271-2](https://doi.org/10.1140/epja/i2016-16271-2)，式 (3)–(5)；本实现只采用其中的三配对相干结构。

ROOT schema 9 记录配置模型、实际模型、网格、接受概率和尝试次数。三路径模型的三颗 α 使用角色 6，`h11b162_labels_are_bookkeeping=1`；`alpha1/2/3` 及 `pair23` 此时仅为存储槽位，不能解释为唯一的先后发射顺序。`h11b162_bose_*_weight` 是任意共同归一化下的振幅强度诊断，已接受事件不能再乘这些权重。

从仓库根目录运行数值验证（Python 需要 NumPy、SciPy）和独立小样本宏：

```bash
cmake --build g4/build --target validate_162_bose -j8
python3 g4/validation/validate_162_bose.py
HB_DATA_DIR=/tmp/11bp-bose-check ./g4/build/HB g4/macros/validation_162_bose_l2.mac 1
```

本地研究集中维护 `jupyter/` 下三份主笔记：

- `codex_0001_p11B三α反应运动学介绍与验证.ipynb`：共用运动学定义后，162 keV 与 675 keV 分节介绍各自的模型、能量预算、图与验证；原角关联及纯 L2 Bose 研究保留在 162 keV 节内。
- `codex_0002_162keV_探测器响应研究.ipynb`：响应一致、固定反应预算、校准失配及有限模板统计，分别分章报告。
- `codex_0003_162keV_弱分支检验与区间校准.ipynb`：零分支基准、边界诊断、区间校准和独立正注入验证。

旧分散笔记及完整计算过程保存在 `jupyter/archive/codex_162keV_20261019/`；历史样本保留原始模型解释。主笔记默认读取已有记录并展示或重绘结果，不自动重跑模拟与大批拟合。旧 `plot_162_primary.py` 和 `plot_162_secondary.py` 会拒绝三路径模型样本，避免把任意槽位误当作物理初级／次级 α；总 α 谱和对称观测量仍可使用。

162 keV 的模型适用性检验位于 `g4/analysis/bayes/gof_162_model_study.py`，结果保存在 0003 第 6.6–6.12 节及图 30–32。它使用独立 BW 伪实验校准联合拟合偏差，再检查四种物理参考；报告率、已报告上限的包含率和失配后不报告约束分别统计。校准限于已声明的固定 MC、经验参考、注入点和响应，不能直接移用到新实验条件。导入及绘图只读缓存；显式 CLI 才启动拟合，已有完成记录时拒绝覆盖。新增单元格的实际 Jupyter 内核验证入口为 `g4/validation/validate_gof_162_notebook.py`，使用 `/home/xc/software/jupyter/.venv/bin/python`，保留历史单元格输出。

Laursen 数据组 9 的条件响应对照位于 `g4/analysis/efficiency/laursen_162_study.py`，结果集中在 0002 第 5 章、图 9–13，模型和推断边界分别见 0001 第 2.8 节、0003 第 7.2 节。它复用选择前反应，计算有限 DSSSD 几何、平均材料能损、触发假设与互斥双重/三重符合；18 组控制尚未同时复现论文的 α₀/α₁ 效率，不能作为真实分支提取的已验证响应。它不替代完整 Geant4 输运或作者的条带匹配算法。

可选目标 `export_162_stopping_power` 导出该研究使用的 Geant4 阻止本领，不运行反应模拟；从根目录配置后以 `cmake --build g4/build --target export_162_stopping_power -j8` 构建。响应测试入口为 `g4/validation/test_laursen_162_response.py`，新增笔记单元格的实际内核验证入口为 `g4/validation/validate_laursen_162_notebooks.py --number 1|2|3`，均使用 `/home/xc/software/jupyter/.venv/bin/python`。日常阅读和绘图只读本地笔记缓存；显式积分 CLI 需要本地 ROOT 和阻止本领输入，并拒绝覆盖已有结果。

重数与条带重叠诊断位于 `g4/analysis/efficiency/laursen_162_readout_audit.py`，结果追加在 0002 第 5.8 节。它重放相同参考，区分单面共条带和同像素，逐阶段核对接受事件总量；没有实现实验匹配算法。仅重分类或共条带拒绝不足以消除当前效率差异，后续需核对分数据组效率的分母、匹配/时间选择及实验通道识别门。对应测试为 `g4/validation/test_laursen_162_readout_audit.py`；新增单元格验证使用上述验证器的 `--number 2 --phase readout`。

学位论文方法的补充检验位于 `g4/analysis/efficiency/laursen_162_selection_audit.py`，结果在 0002 第 5.9 节、图 14。它补入带版本标记的组 9 效率和相对能分类门，对显式探测器平面方程、共面性编号、硬阈值及附加选择作九组控制；保留旧响应结果。最高能共面性规则及通道分类通过存储置换检查，但完整条带匹配仍未实现，九组条件未同时复现外部效率。下一项是实验版本的完整几何、逐条带标定、匹配及生成/观测计数的逐阶段核对。测试入口为 `g4/validation/test_laursen_162_selection_audit.py`，实际内核验证用 `--number 1|2|3 --phase thesis`；导入及绘图只读缓存，显式重放需要已核指纹的学位论文 PDF 和本地反应库。

条带信号匹配控制位于 `g4/analysis/efficiency/laursen_162_strip_matching.py`，全样本入口为 `laursen_162_matching_study.py`，结果在 0002 第 5.10 节、图 15–16。匹配只使用条带编号与标定能量，分别检验求和优先、共享优先和拒绝歧义；七组控制保存重数迁移、通道混淆和 D1 能谱。理想电荷共享被误解为两颗粒子时可产生虚假的基态识别，不能用真值配对或公共效率系数代替该过程。此为独立算法控制，历史作者匹配与硬件标定仍未取得，外部效率尚未同时复现。当前没有额外重建总能量/激发能窗，后续先检查错误匹配的能量闭合，再核对完整接受规则。测试为 `test_laursen_162_strip_matching.py` 和 `test_laursen_162_matching_study.py`；实际内核验证使用 `--number 1|2|3 --phase matching`。默认读取本地缓存，显式 `--output` 才计算并拒绝覆盖完成结果。

675 keV 的 100 万入射样本保存在本地 `g4/data/spectrum_675_lab_coherent_L1L3_1M_20261020.root`。只读全样本验证入口为 `g4/validation/validate_675_generator.py`；运行条件、种子、验证结果和图 12 集中记录在 0001 的第 3.3 节，原 100 事件回归记录保留在第 3.2 节。

## 编译与运行

需要 CMake、C++17 编译器、Geant4、ROOT，以及合并线程输出所用的 `hadd`。整理时本机环境为 Geant4 11.3.2、ROOT 6.34.10；应先加载安装环境，让 CMake 能找到二者。

在仓库根目录执行：

```bash
cmake -S g4 -B g4/build -DCMAKE_BUILD_TYPE=Release
cmake --build g4/build -j8
cd g4
./batch.sh macros/run.mac 8
```

脚本默认 8 个工作线程，允许 1–8；从仓库根目录也可运行 `./g4/batch.sh g4/macros/run.mac 8`。数据写入 `g4/data/`，线程输出成功合并后默认清理线程文件；`CLEAN_THREADS=0` 可保留它们。同一数据目录应按能量点顺序使用，避免同时运行多个 `batch.sh`。

旧构建缓存含原目录的绝对路径，迁移时保存在本地归档；应在 `g4/build/` 重新配置和编译。最早的历史标签 `g4_251202` 仍使用 `B2a/` 和 `exampleB2a`，不能直接套用当前 HB 命令。

## α 能谱工作流

以下命令从 `g4/` 执行。Python 分析使用 NumPy、Matplotlib、uproot；部分探测器分析还使用 Awkward Array。ROOT 数据不随仓库分发。

```bash
# 162 keV 单点：仅启用 162 分量，100 万个入射事件
./batch.sh macros/spectrum_162_lab_05_95_00.mac 8

# 150–300 keV、步长 10 keV，每点 100 万个入射事件
python3 analysis/alpha_spectrum/run_scan.py

# 50–200 keV，各点单独绘图与总览
python3 analysis/alpha_spectrum/run_scan.py --energies 50 200 10 --separate
python3 analysis/alpha_spectrum/plot_lab_total.py --energies 50 200 10 --separate --overview
```

`run_scan.py` 校验、复用已有单点数据，并按约定写入 `ScanInfo`。独立运行 `batch.sh` 的输出带时间戳；162 单点绘图默认读取 `data/spectrum_162_lab_05_95_00.root`，也可显式传入合并 ROOT 文件：

```bash
python3 analysis/alpha_spectrum/plot_162_lab_total.py data/your_merged_output.root --total-only
python3 analysis/alpha_spectrum/plot_lab_total.py --total-only
```

每次反应的三颗 α 刚产生时的 LAB 动能分别填入同一张直方图，全立体角、每颗计一次。默认 **25 keV/bin**，横轴 MeV，纵轴 Counts；162 keV 单点范围 0–6.5 MeV，扫描谱范围 0–7 MeV。多能量点合并采用计数直接相加，不额外乘截面或束流权重。

能谱宏使用 `0.12 um` 的 `Enriched_B11` 靶、无背衬、点束流和 `1e11` 截面偏置。162 分量的基态／2⁺／直接分支设为 5%／95%／0%；扫描宏同时启用 675 分量，该分量取 0%／100%／0%。这些是当前样本配置，具体以宏和 ROOT 保存的信息为准。

## 查看与回溯版本

```bash
git log --oneline --decorate --all
git tag --list 'g4_*'
git show g4_260921
git diff g4_260629 g4_260714 -- g4/src g4/include g4/macros

# 在独立工作区查看旧版本，保留当前目录的数据
git worktree add --detach ../11B_p_260629 g4_260629
```

11 个日期标签保存对应目录的源码快照。标签日期来自目录名称，部分目录在该日期之后继续修改过；提交说明记录原 Git HEAD 和是否存在本地修改。所有迁移提交使用实际整理时间。无日期的 `xc/` 单独保存在 `g4_xc_legacy` 标签，不假定其开发日期。

原目录、原 Git 元数据和历史构建产物保存在本机 `.local_archive/unify_20260923/`，不上传。当前本地 ROOT、图像、TXT 和 `Note.ipynb` 随最新工程保留在 `g4/`。标签用于还原公开源码；完整运行现场还需相应的本地归档和 ROOT 数据。

## 验证与已知限制

整理时逐文件核对了所有快照的内容与可执行权限，并检查提交历史中的排除规则。迁移适配只调整 `batch.sh` 的工程目录识别，使它在外层统一 Git 仓库中定位 `g4/`；历史标签保留原脚本。其余物理源码按来源保存。

2026-09-23 在新路径完成 CMake 配置和 `-j8` 编译；Bash 与当前全部 Python 文件语法检查通过，批处理目录定位在仓库根目录、`g4/`、`g4/build/` 三种启动位置通过。以 162 keV 能谱宏临时缩减为 20 个入射事件、2 线程运行，确认日志完成 20 个事件、ROOT 含 20 次反应和 60 颗 α，事件号覆盖 0–19，能量守恒残差的展宽为 `3.92e-12 MeV`。测试宏和数据写在 `/tmp/`，未重跑正式百万事件样本。

历史版本未逐一重新编译或重跑大规模模拟。当前 `HB.cc` 尚未将 Geant4 宏执行错误转换为非零进程退出码，`batch.sh` 在无新 ROOT 文件时也可能仅警告并返回成功；生产计算应检查 Geant4 日志与实际完成事件数。这是后续应优先修复的运行可靠性问题。

后续修改建议按物理模型、几何、ROOT 数据结构、分析分开提交；重要计算使用有说明的标签，在本地运行记录中保留提交号、宏、随机种子、入射事件数及数据文件名。

## 许可证

保留原仓库的 [MIT License](LICENSE)。源文件中已有的 Geant4 等组件版权与许可声明一并保留。
