import csv
import math
import random
import matplotlib.pyplot as plt

def add_noise(data, noise_level):
    return [y + random.uniform(-noise_level, noise_level) for y in data]
def read_csv(file_path):

    x_values = []
    y_values = []
    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        if file_path == "MountEverest.csv":
            next(csvreader)
        for row in csvreader:
            x_values.append(float(row[0]))
            y_values.append(float(row[1]))

    return x_values, y_values

def chebyshev_nodes(a, b, n):
    nodes = []
    for i in range(1, n + 1):
        x_i = 0.5 * (a + b) + 0.5 * (b - a) * math.cos((2 * i - 1) * math.pi / (2 * n))
        nodes.append(x_i)
    return nodes


def lagrange_interpolation(x_values, y_values, x):
    def basis_polynomial(j, x):
        p = 1
        for i in range(len(x_values)):
            if i != j:
                p *= (x - x_values[i]) / (x_values[j] - x_values[i])
        return p

    n = len(x_values)
    y_interpolated = 0
    for j in range(n):
        y_interpolated += y_values[j] * basis_polynomial(j, x)
    return y_interpolated


def plot_lagrange_interpolation(y_interpolated, x_values, y_values, n,x_nodes,y_nodes,additional_info=""):

    # Plot on the first subplot
    plt.plot(x_values, y_values, label='Original Data')
    plt.title('Lagrange Interpolation: '+ str(n)+ ' nodes '+additional_info)
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.legend()
    plt.grid(True)

    # Plot on the second subplot
    plt.plot(x_values, y_interpolated, label='Interpolated Data', color='orange')
    plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label='Interpolation Nodes')

    plt.legend()
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Show the figure
    plt.show()


def cubic_spline_interpolation(x_values, y_values, x):
    n = len(x_values)
    a = y_values.copy()

    h = [x_values[i + 1] - x_values[i] for i in range(n - 1)]
    alpha = [(3/h[i]) * (a[i + 1] - a[i]) - (3/h[i - 1]) * (a[i] - a[i - 1]) for i in range(1, n - 1)]

    l = [1] * n
    mu = [0] * n
    z = [0] * n

    for i in range(1, n - 1):
        l[i] = 2 * (x_values[i + 1] - x_values[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i]

    b = [0] * n
    c = [0] * n
    d = [0] * n

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    for i in range(n - 1):
        if x_values[i] <= x <= x_values[i + 1]:
            dx = x - x_values[i]
            return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

def plot_cubic_spline_interpolation(x_values, y_values,y_interpolated,n, x_nodes,y_nodes):
    plt.plot(x_values, y_values, label='Original Data')
    plt.plot(x_values, y_interpolated, label='Interpolated Data')
    plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label='Interpolation Nodes')
    plt.title('Cubic Spline Interpolation: ' + str(n) + ' nodes')
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    file_path = '100.csv'
    file_path2='MountEverest.csv'
    x_values, y_values = read_csv(file_path)
    x_values_ev,y_values_ev = read_csv(file_path2)

    print("CSV has been read successfully")
    x_nodes_5 = x_values[1::len(x_values)//5]
    y_nodes_5 = y_values[1::len(y_values)//5]
    x_nodes_10 = x_values[1::len(x_values)//10]
    y_nodes_10 = y_values[1::len(y_values)//10]
    x_nodes_15=x_values[1::len(x_values)//15]
    y_nodes_15 = y_values[1::len(y_values) // 15]
    x_nodes_20 = x_values[1::len(x_values) // 20]
    y_nodes_20 = y_values[1::len(y_values) // 20]
    y_interpolated_5 = []
    y_interpolated_10 = []
    y_interpolated_15=[]
    y_interpolated_20=[]

    for x in x_values:
        y_interpolated_5.append(lagrange_interpolation(x_nodes_5, y_nodes_5, x))
        y_interpolated_10.append(lagrange_interpolation(x_nodes_10, y_nodes_10, x))
        y_interpolated_15.append(lagrange_interpolation(x_nodes_15, y_nodes_15, x))
        y_interpolated_20.append(lagrange_interpolation(x_nodes_20, y_nodes_20, x))

    plot_lagrange_interpolation(y_interpolated_5, x_values, y_values,5,x_nodes_5,y_nodes_5)
    plot_lagrange_interpolation(y_interpolated_10, x_values, y_values,10,x_nodes_10,y_nodes_10)
    plot_lagrange_interpolation(y_interpolated_15, x_values, y_values,15,x_nodes_15,y_nodes_15)
    plot_lagrange_interpolation(y_interpolated_20, x_values, y_values,20,x_nodes_20,y_nodes_20)

    y_interpolated_5 = [cubic_spline_interpolation(x_nodes_5, y_nodes_5, x) for x in x_values]
    y_interpolated_10 = [cubic_spline_interpolation(x_nodes_10, y_nodes_10, x) for x in x_values]
    y_interpolated_15 = [cubic_spline_interpolation(x_nodes_15, y_nodes_15, x) for x in x_values]
    y_interpolated_20 = [cubic_spline_interpolation(x_nodes_20, y_nodes_20, x) for x in x_values]
    plot_cubic_spline_interpolation(x_values, y_values, y_interpolated_5,5,x_nodes_5,y_nodes_5)
    plot_cubic_spline_interpolation(x_values, y_values, y_interpolated_10,10,x_nodes_10,y_nodes_10)
    plot_cubic_spline_interpolation(x_values, y_values, y_interpolated_15,15,x_nodes_15,y_nodes_15)
    plot_cubic_spline_interpolation(x_values, y_values, y_interpolated_20,20,x_nodes_20,y_nodes_20)

    x_nodes_5 = x_values_ev[1::len(x_values_ev) // 5]
    y_nodes_5 = y_values_ev[1::len(y_values_ev) // 5]
    x_nodes_10 = x_values_ev[1::len(x_values_ev) // 10]
    y_nodes_10 = y_values_ev[1::len(y_values_ev) // 10]
    x_nodes_15 = x_values_ev[1::len(x_values_ev) // 15]
    y_nodes_15 = y_values_ev[1::len(y_values_ev) // 15]
    x_nodes_20 = x_values_ev[1::len(x_values_ev) // 20]
    y_nodes_20 = y_values_ev[1::len(y_values_ev) // 20]

    y_interpolated_5 = []
    y_interpolated_10 = []
    y_interpolated_15=[]
    y_interpolated_20=[]

    for x in x_values_ev:
        y_interpolated_5.append(lagrange_interpolation(x_nodes_5, y_nodes_5, x))
        y_interpolated_10.append(lagrange_interpolation(x_nodes_10, y_nodes_10, x))
        y_interpolated_15.append(lagrange_interpolation(x_nodes_15, y_nodes_15, x))
        y_interpolated_20.append(lagrange_interpolation(x_nodes_20, y_nodes_20, x))

    plot_lagrange_interpolation(y_interpolated_5, x_values_ev, y_values_ev,5,x_nodes_5,y_nodes_5)
    plot_lagrange_interpolation(y_interpolated_10, x_values_ev, y_values_ev,10,x_nodes_10,y_nodes_10)
    plot_lagrange_interpolation(y_interpolated_15, x_values_ev, y_values_ev,15,x_nodes_15,y_nodes_15)
    plot_lagrange_interpolation(y_interpolated_20, x_values_ev, y_values_ev,20,x_nodes_20,y_nodes_20)

    y_interpolated_5 = [cubic_spline_interpolation(x_nodes_5, y_nodes_5, x) for x in x_values_ev]
    y_interpolated_10 = [cubic_spline_interpolation(x_nodes_10, y_nodes_10, x) for x in x_values_ev]
    y_interpolated_15 = [cubic_spline_interpolation(x_nodes_15, y_nodes_15, x) for x in x_values_ev]
    y_interpolated_20 = [cubic_spline_interpolation(x_nodes_20, y_nodes_20, x) for x in x_values_ev]
    plot_cubic_spline_interpolation(x_values_ev, y_values_ev, y_interpolated_5,5,x_nodes_5,y_nodes_5)
    plot_cubic_spline_interpolation(x_values_ev, y_values_ev, y_interpolated_10,10,x_nodes_10,y_nodes_10)
    plot_cubic_spline_interpolation(x_values_ev, y_values_ev, y_interpolated_15,15,x_nodes_15,y_nodes_15)
    plot_cubic_spline_interpolation(x_values_ev, y_values_ev, y_interpolated_20,20,x_nodes_20,y_nodes_20)

    y_interpolated_5 = []
    y_interpolated_10 = []
    y_interpolated_15=[]
    y_interpolated_20=[]
    x_nodes_5=chebyshev_nodes(x_values[0],x_values[-1],5)
    y_nodes_5=[]
    for j in range(len(x_nodes_5)):
        for i in range(len(x_values)):
            if x_nodes_5[j]<=x_values[i]:
                y_nodes_5.append(y_values[i])
                break



    x_nodes_10=chebyshev_nodes(x_values[0],x_values[-1],10)
    y_nodes_10=[]
    for j in range(len(x_nodes_10)):
        for i in range(len(x_values)):
            if x_nodes_10[j]<=x_values[i]:
                y_nodes_10.append(y_values[i])
                break


    x_nodes_15=chebyshev_nodes(x_values[0],x_values[-1],15)
    y_nodes_15=[]
    for j in range(len(x_nodes_15)):
        for i in range(len(x_values)):
            if x_nodes_15[j]<=x_values[i]:
                y_nodes_15.append(y_values[i])
                break

    x_nodes_20=chebyshev_nodes(x_values[0],x_values[-1],20)
    y_nodes_20=[]
    for j in range(len(x_nodes_20)):
        for i in range(len(x_values)):
            if x_nodes_20[j]<=x_values[i]:
                y_nodes_20.append(y_values[i])
                break

    for x in x_values:
        y_interpolated_5.append(lagrange_interpolation(x_nodes_5, y_nodes_5, x))
        y_interpolated_10.append(lagrange_interpolation(x_nodes_10, y_nodes_10, x))
        y_interpolated_15.append(lagrange_interpolation(x_nodes_15, y_nodes_15, x))
        y_interpolated_20.append(lagrange_interpolation(x_nodes_20, y_nodes_20, x))

    plot_lagrange_interpolation(y_interpolated_5, x_values, y_values,5,x_nodes_5,y_nodes_5, "Chebushev Nodes")
    plot_lagrange_interpolation(y_interpolated_10, x_values, y_values,10,x_nodes_10,y_nodes_10, "Chebushev Nodes")
    plot_lagrange_interpolation(y_interpolated_15, x_values, y_values,15,x_nodes_15,y_nodes_15, "Chebushev Nodes")
    plot_lagrange_interpolation(y_interpolated_20, x_values, y_values,20,x_nodes_20,y_nodes_20, "Chebushev Nodes")


    y_interpolated_5 = []
    y_interpolated_10 = []
    y_interpolated_15=[]
    y_interpolated_20=[]
    x_nodes_5=chebyshev_nodes(x_values[0],x_values[-1],5)
    y_nodes_5=[]
    for j in range(len(x_nodes_5)):
        for i in range(len(x_values)):
            if x_nodes_5[j]<=x_values[i]:
                y_nodes_5.append(y_values[i])
                break



    x_nodes_10=chebyshev_nodes(x_values[0],x_values[-1],10)
    y_nodes_10=[]
    for j in range(len(x_nodes_10)):
        for i in range(len(x_values)):
            if x_nodes_10[j]<=x_values[i]:
                y_nodes_10.append(y_values[i])
                break


    x_nodes_15=chebyshev_nodes(x_values[0],x_values[-1],15)
    y_nodes_15=[]
    for j in range(len(x_nodes_15)):
        for i in range(len(x_values)):
            if x_nodes_15[j]<=x_values[i]:
                y_nodes_15.append(y_values[i])
                break

    x_nodes_20=chebyshev_nodes(x_values[0],x_values[-1],20)
    y_nodes_20=[]
    for j in range(len(x_nodes_20)):
        for i in range(len(x_values)):
            if x_nodes_20[j]<=x_values[i]:
                y_nodes_20.append(y_values[i])
                break


    for x in x_values:
        y_interpolated_5.append(lagrange_interpolation(x_nodes_5, y_nodes_5, x))
        y_interpolated_10.append(lagrange_interpolation(x_nodes_10, y_nodes_10, x))
        y_interpolated_15.append(lagrange_interpolation(x_nodes_15, y_nodes_15, x))
        y_interpolated_20.append(lagrange_interpolation(x_nodes_20, y_nodes_20, x))

    y_interpolated_5=add_noise(y_interpolated_5,3)
    y_interpolated_10 = add_noise(y_interpolated_10, 3)
    y_interpolated_15 = add_noise(y_interpolated_15, 3)
    y_interpolated_20 = add_noise(y_interpolated_20, 3)

    plot_lagrange_interpolation(y_interpolated_5, x_values, y_values,5,x_nodes_5,y_nodes_5,"Chebushev+Noise")
    plot_lagrange_interpolation(y_interpolated_10, x_values, y_values,10,x_nodes_10,y_nodes_10,"Chebushev+Noise")
    plot_lagrange_interpolation(y_interpolated_15, x_values, y_values,15,x_nodes_15,y_nodes_15,"Chebushev+Noise")
    plot_lagrange_interpolation(y_interpolated_20, x_values, y_values,20,x_nodes_20,y_nodes_20,"Chebushev+Noise")


if __name__ == "__main__":
    main()
