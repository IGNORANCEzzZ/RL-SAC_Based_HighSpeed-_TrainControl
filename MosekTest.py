import cvxpy as cp
import numpy as np

# 1. 检查 CVXPy 是否能找到 MOSEK
print("CVXPy aнализ установленных решателей:", cp.installed_solvers())
if 'MOSEK' not in cp.installed_solvers():
    print("\n错误：CVXPy 未能找到 MOSEK！")
    print("请确认 MOSEK 和 CVXPy 安装在同一个 Python 环境中。")
else:
    print("\n成功：CVXPy 找到了 MOSEK 求解器！")

    # 2. 定义一个简单的优化问题
    # 创建两个变量
    x = cp.Variable()
    y = cp.Variable()

    # 创建约束条件
    constraints = [x + y == 1,
                   x - y >= 1]

    # 创建目标函数
    obj = cp.Minimize((x - y)**2)

    # 定义问题
    prob = cp.Problem(obj, constraints)

    # 3. 使用 MOSEK 求解器求解问题
    print("\n尝试使用 MOSEK 求解...")
    try:
        prob.solve(solver=cp.MOSEK, verbose=True) # verbose=True 会打印求解器日志

        # 4. 打印结果
        print("\n求解完成！")
        print("问题状态:", prob.status)
        print("最优值:", prob.value)
        print("x 的最优解:", x.value)
        print("y 的最优解:", y.value)
    except Exception as e:
        print(f"\n使用 MOSEK 求解时发生错误: {e}")