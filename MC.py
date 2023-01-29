import numpy as np
from math import exp, sqrt
import pandas as pd
import matplotlib.pyplot as plt

# 构造一个随机数算法
def box_muller(m, n, lower=0, upper=1):
    Z = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            u = 1 - np.random.uniform(lower, upper)
            v = np.random.uniform(lower, upper)
            Z[i, j] = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
    return Z

class MC:

    # 初始化变量
    def __init__(self, S_0, T1, T2, alpha, r, q, sigma):
        self.S_0 = S_0
        self.T1 = T1/360
        self.T2 = T2/360
        self.alpha = alpha
        # 用t=0的r,q,sigma代替t=0到T2时间段的r,q,sigma
        self.r = r
        self.mu = r - q
        self.sigma = sigma 

    # 考虑两阶段的蒙特卡洛模拟, 0-T1, T1-T2
    # step1: 模拟出0-T1，T1-T2的股价走势

    # 考虑m次n步的蒙特卡洛模拟

    def MC_GBM(self, S_t1, t1, t2, m, n, Z_arr=np.zeros(1)):
        mu, sigma = self.mu, self.sigma
        if Z_arr.all() == 0:
            Z_arr = box_muller(m, n)
        S_t_arr = np.zeros((m, n+1)) 
        dt = (t2 - t1)/n
        p = sigma * sqrt(dt)
        for i in range(m):
            S_t_arr[i, 0] = S_t1
            for j in range(n):
                S_t_arr[i, j+1] = S_t_arr[i, j] * (1 + mu * dt + p * Z_arr[i, j] + 0.5 * sigma ** 2 * dt * (Z_arr[i, j]**2-1)\
                    + 0.5 * mu ** 2 * dt + mu * p * dt * Z_arr[i, j]) # 离散化方法：second order
        return S_t_arr 
    
    # step2: 计算出欧式看涨期权在t=T1的期权价格
    def call_T1(self, S_T1, S_T2_arr, m2):
        alpha, r, T1, T2 = self.alpha, self.r, self.T1, self.T2

        C_arr = np.zeros(m2)
        K = alpha * S_T1
        for i in range(m2):
            if S_T2_arr[i] > K:
                C_arr[i] = exp(-r*(T2-T1))*(S_T2_arr[i] - K)
            else:
                C_arr[i] = 0
        return C_arr

    # step3: 根据欧式看涨期权在t=T1的期权价格计算出X的价格
    def X_0(self, C_T1):
        r, T1 = self.r, self.T1
        X_0 = exp(-r*T1)*C_T1
        return X_0

    def get_solution(self, m1, m2, n1, n2, Z_arr1=np.zeros(1), Z_arr2=np.zeros(1)):
        S_0, T1, T2 = self.S_0, self.T1, self.T2
        Stage1_arr = self.MC_GBM(S_0, 0, T1, m1, n1, Z_arr=Z_arr1)
        X_0_arr = np.zeros(m1)
        Stage2_arr = np.zeros((m2, n2+1, m1))
        C_T1_arr = np.zeros(m1)
        for i in range(m1):
            S_T1 = Stage1_arr[i, -1]
            Stage2_arr[:, :, i] = self.MC_GBM(S_T1, T1, T2, m2, n2, Z_arr=Z_arr2)
            C_T1_arr[i] = np.mean(self.call_T1(S_T1, Stage2_arr[:, -1, i], m2)) # 根据t=T1,T2的股票价格得到t=T1的期权价格的估计值
            X_0_arr[i] = self.X_0(C_T1_arr[i]) # 将期权价格折现到t=0
        return Stage1_arr, Stage2_arr, X_0_arr, C_T1_arr
    
def Visualization():
    def Initial(S_0=157.96, T1=15, T2=78, alpha=1, r=0.0351, q=0.0015, sigma=0.2292, m1=400, m2=400, n1=8, n2=8):
        test = MC(S_0, T1, T2, alpha, r, q, sigma)
        solution = test.get_solution(m1, m2, n1, n2)
        return solution

    Initial_test = Initial()

    def S_t_line(solution):
        plt.figure(figsize=(20,8), dpi=80)
        step1 = np.arange(0, 9)
        step2 = np.arange(8, 17)
        Stage1_arr = solution[0]
        Stage2_arr = solution[1]
        for i in range(400):
            plt.plot(step1, Stage1_arr[i, :])
            for j in range(400):
                plt.plot(step2, Stage2_arr[j, :, i])
            
        plt.title("MC(n steps)")
        plt.xlabel("step")
        plt.ylabel("S_t")
        plt.savefig('Visual_MC.jpg')
    
    S_t_line(Initial_test)

    def price_hist(S, name):
        plt.figure()
        plt.hist(S, 50, density=True)
        plt.title("Distribution of " + name)
        plt.xlabel(name)
        plt.ylabel("Frequency")
        plt.savefig("Distribution of " + name + ".jpg")

    price_hist(Initial_test[0][:, -1], 'S_T1')
    price_hist(Initial_test[1][:, -1, :], 'S_T2')
    price_hist(Initial_test[2], 'X_0')

# 利用方差缩减技术减小方差

class MC_dual(MC):

    # 对偶变量法
    def get_solution_dual(self, m1, m2, n1, n2):
        Z_arr1 = box_muller(m1, n1)
        Z_arr_dual1 = - Z_arr1
        Z_arr2 = box_muller(m2, n2)
        Z_arr_dual2 = - Z_arr2

        X_arr = self.get_solution(m1, m2, n1, n2, Z_arr1, Z_arr2)[2]
        Y_arr = self.get_solution(m1, m2, n1, n2, Z_arr_dual1, Z_arr_dual2)[2]
        X_arr_dual = (X_arr + Y_arr)/2
        return X_arr_dual

class MC_control(MC): 
    # 控制变量法
    # 生成控制变量
    def get_solution_control(self, m1, m2, n1, n2, X_arr=np.zeros(1), Control_arr1=np.zeros(1), Control_arr2=np.zeros(1)):
        if X_arr.all() == 0 and Control_arr1.all() == 0 and Control_arr2.all() == 0:
            result = self.get_solution(m1, m2, n1, n2)
            X_arr = result[2]

            Control_arr1 = np.zeros(m1)
            for i in range(m1):
                Control_arr1[i] = np.mean(result[1][:, -1, i])

            Control_arr2 = result[0][:, -1]

        beta1 = (np.cov(X_arr, Control_arr1)[0, 1])/np.var(Control_arr1, ddof=1)
        beta2 = (np.cov(X_arr, Control_arr2)[0, 1])/np.var(Control_arr2, ddof=1)
        X_arr_control = np.zeros(m1)
        for i in range(m1):
            X_arr_control[i] = X_arr[i] - beta1*(Control_arr1[i] - np.mean(Control_arr1)) - beta2*(Control_arr2[i] - np.mean(Control_arr2))
        return X_arr_control

class MC_dual_control(MC_control): 
    # 对偶变量+控制变量
    def get_solution_dual_control(self, m1, m2, n1, n2):
        Z_arr1 = box_muller(m1, n1)
        Z_arr_dual1 = - Z_arr1
        Z_arr2 = box_muller(m2, n2)
        Z_arr_dual2 = - Z_arr2
        result = self.get_solution(m1, m2, n1, n2, Z_arr1, Z_arr2)
        result_dual = self.get_solution(m1, m2, n1, n2, Z_arr_dual1, Z_arr_dual2)
        X_arr_dual = (result[2] + result_dual[2])/2
        Control_arr1 = np.zeros(m1)
        for i in range(m1):
            Control_arr1[i] = np.mean(result[1][:, -1, i] + result_dual[1][:, -1, i])
        Control_arr2 = (result[0][:, -1] + result_dual[0][:, -1])/2

        result_dual_control = super().get_solution_control(m1, m2, n1, n2, X_arr_dual, Control_arr1, Control_arr2)

        return result_dual_control

def variance_loss():
    var = MC(157.96, 15, 78, 1, 0.0351, 0.0015, 0.2292) # 类的实例化

    result0 = var.get_solution(400, 400, 8, 8)[2]
    print("Variance of MC:",np.var(result0, ddof=1))

    var_dual = MC_dual(157.96, 15, 78, 1, 0.0351, 0.0015, 0.2292)
    result1 = var_dual.get_solution_dual(400, 400, 8, 8)
    print("Variance of MC_dual:", np.var(result1, ddof=1))

    var_control = MC_control(157.96, 15, 78, 1, 0.0351, 0.0015, 0.2292)
    result2 = var_control.get_solution_control(400, 400, 8, 8)
    print("Variance of MC_control:", np.var(result2, ddof=1))

    var_dual_control = MC_dual_control(157.96, 15, 78, 1, 0.0351, 0.0015, 0.2292)
    result3 = var_dual_control.get_solution_dual_control(400, 400, 8, 8)

    print("Variance of MC_dual_control:", np.var(result3, ddof=1))


def Simulation():
    # 以2022-09-01作为测试日期
    def basic(S_0=157.96, T1=15, T2=78, alpha=1, r=0.0351, q=0.0015, sigma=0.2292, m1=400, m2=400, n1=8, n2=8):
        test = MC_dual_control(S_0, T1, T2, alpha, r, q, sigma)
        solution = np.mean(test.get_solution_dual_control(m1, m2, n1, n2))
        return solution

    basic_test = basic()

    def basic_std(S_0=157.96, T1=15, T2=78, alpha=1, r=0.0351, q=0.0015, sigma=0.2292, m1=400, m2=400, n1=8, n2=8):
        test = MC_dual_control(S_0, T1, T2, alpha, r, q, sigma)
        solution = np.std(test.get_solution_dual_control(m1, m2, n1, n2), ddof=1)
        return solution
    basic_test_std = basic_std()
    print('95% confidence interval:', [basic_test-1.96*basic_test_std, basic_test+1.96*basic_test_std])
  
    # 改变模拟次数
    def change_m(m):
        m_test = basic(m1=m, m2=m)
        return m_test

    def draw_1():
        m_arr = [50, 100, 150, 200, 250, 300, 350, 400]
        solution = []
        for m in m_arr:
            solution.append(change_m(m))
        plt.figure()
        plt.plot(m_arr, solution)
        plt.title('Effect of paths')
        plt.xlabel("paths")
        plt.ylabel("price of X(t=0)")
        plt.savefig('Simul_1.jpg')

    draw_1()

    def change_n(n):
        n_test = basic(n1=n, n2=n)
        return n_test

    def draw_2():
        n_arr = [1, 2, 3, 4, 5, 6, 7, 8]
        solution = []
        for n in n_arr:
            solution.append(change_n(n))
        plt.figure()
        plt.plot(n_arr, solution)
        plt.title('Effect of steps')
        plt.xlabel("steps")
        plt.ylabel("price of X(t=0)")
        plt.savefig('Simul_2.jpg')
    
    draw_2()

    # 改变alpha

    def draw_3():
        alpha_test_1 = basic(alpha = 0.99)
        alpha_test_2 = basic(alpha = 1.01)
        x = ['0.99', '1', '1.01']
        y = [alpha_test_1, basic_test, alpha_test_2]
        plt.figure()
        plt.bar(x, y, width=0.5)
        plt.title('Effect of alpha')
        plt.xlabel("alpha")
        plt.ylabel("price of X(t=0)")
        plt.savefig('Simul_3.jpg')

    draw_3()

    # 在此基础上，固定T1, 改变T2——改变现货价格。利率,股息,波动率也要相应地改变——衡量X的标的期权与其剩余到期期限的关系
    df = pd.read_csv('data.csv') # 读取每列数据
    dates = pd.to_datetime(df['date']) # 转换为datetime格式
    df['date'] = dates

    def change_T2(delay, alpha=1):
        
        solution = []
        date_solution = []
        for i in delay:
            T2 = df.loc[i, 'T']
            S_0 = df.loc[i, 'stock_price']
            r = float(df.loc[i, 'r'].strip('%'))/100 # 数据格式转换
            q = float(df.loc[i, 'q'].strip('%'))/100 
            sigma = float(df.loc[i, 'sigma'].strip('%'))/100
            solution.append(basic(T2=T2, S_0=S_0, r=r, q=q, sigma=sigma, alpha=alpha))
            date_solution.append(df.loc[i, 'date'])
        return date_solution, solution

    def draw_4():
        delay = [5, 10, 15, 20, 25, 30, 35, 40]  # T1=15,所以T2需要大于15,delay小于43 
        solution1 = change_T2(delay, 0.99)
        solution2 = change_T2(delay)
        solution3 = change_T2(delay, 1.01)
        date = solution1[0]
        plt.figure(figsize=(20,8), dpi=80)
        plt.plot(date, solution1[1], c='green', label='alpha=0.99')
        plt.plot(date, solution2[1], label='alpha=1')
        plt.plot(date, solution3[1], c='red', label='alpha=1.01')
        plt.legend()
        plt.title('Effect of T2 with fixed T1')
        plt.xlabel("spot date(t=0)")
        plt.ylabel("price of X(t=0)")
        plt.savefig('Simul_4.jpg')
        
    draw_4()

    # 固定T2-T1, 改变T1——衡量X与剩余到期期限T1的关系

    def change_T2_1(delay, alpha=1):
        solution = []
        date_solution = []
        for i in delay:
            T2 = df.loc[i, 'T']
            T1 = T2 - 63 # 原来是78-63
            S_0 = df.loc[i, 'stock_price']
            r = float(df.loc[i, 'r'].strip('%'))/100 # 数据格式转换
            q = float(df.loc[i, 'q'].strip('%'))/100 
            sigma = float(df.loc[i, 'sigma'].strip('%'))/100
            solution.append(basic(T1=T1, T2=T2, S_0=S_0, r=r, q=q, sigma=sigma, alpha=alpha))
            date_solution.append(df.loc[i, 'date'])
        return date_solution, solution

    def draw_5():
        delay = [1, 2, 3, 4, 5, 6, 7, 8]  # T1需要小于15,delay小于10 
        solution1 = change_T2_1(delay, 0.99)
        solution2 = change_T2_1(delay)
        solution3 = change_T2_1(delay, 1.01)
        date = solution1[0]
        plt.figure(figsize=(20,8), dpi=80)
        plt.plot(date, solution1[1], c='green', label='alpha=0.99')
        plt.plot(date, solution2[1], label='alpha=1')
        plt.plot(date, solution3[1], c='red', label='alpha=1.01')
        plt.legend()
        plt.title('Effect of T1 with fixed T2-T1')
        plt.xlabel("spot date(t=0)")
        plt.ylabel("price of X(t=0)")
        plt.savefig('Simul_5.jpg')

    draw_5()

    # 固定T2，改变T1——现货价格不变
    
    def change_T1_2(T1_list, alpha=1):
        solution = []
        for T1 in T1_list:
            solution.append(basic(T1=T1, alpha=alpha))
        return solution

    def draw_6():
        T1_list = [1, 5, 15, 20, 40, 50, 70] # 对应T=77,73,63,58,38,28,8

        T1_alpha_test_1 = change_T1_2(T1_list=T1_list, alpha = 0.99)
        T1_alpha_test_2 = change_T1_2(T1_list=T1_list)
        T1_alpha_test_3 = change_T1_2(T1_list=T1_list, alpha = 1.01)

        plt.figure(figsize=(20,8), dpi=80)
        plt.plot(T1_list, T1_alpha_test_1, c='green', label='alpha=0.99')
        plt.plot(T1_list, T1_alpha_test_2, label='alpha=1')
        plt.plot(T1_list, T1_alpha_test_3, c='red', label='alpha=1.01')
        plt.legend()
        plt.title('Effect of T1 with fixed T2')
        plt.xlabel("T1")
        plt.ylabel("price of X(t=0)")
        plt.savefig('Simul_6.jpg')

    draw_6()

def main():
    Visualization()
    variance_loss()
    Simulation()
    return 0

if __name__ == "__main__":
    main()


