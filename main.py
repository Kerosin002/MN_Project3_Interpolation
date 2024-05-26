import csv
import matplotlib.pyplot as plt


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


def plot_lagrange_interpolation(y_interpolated_20, y_interpolated_5, x_values, y_values):

    # Plot on the first subplot
    plt.plot(x_values, y_values, label='Original Data')
    plt.title('Lagrange Interpolation')
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.legend()
    plt.grid(True)

    # Plot on the second subplot
    plt.plot(x_values, y_interpolated_5, label='Interpolated Data 5 nodes', color='orange')
    plt.plot(x_values, y_interpolated_20, label='Interpolated Data 20 nodes', color='red')

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

def plot_cubic_spline_interpolation(x_values, y_values, y_interpolated_5, y_interpolated_20):
    plt.plot(x_values, y_values, label='Original Data')
    plt.plot(x_values, y_interpolated_5, label='Interpolated Data 5 nodes')
    plt.plot(x_values, y_interpolated_20, label='Interpolated Data 20 nodes')
    plt.title('Cubic Spline Interpolation')
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
    x_nodes_20 = x_values[1::len(x_values)//10]
    y_nodes_20 = y_values[1::len(y_values)//10]
    y_interpolated_5 = []
    y_interpolated_20 = []

    for x in x_values:
        y_interpolated_5.append(lagrange_interpolation(x_nodes_5, y_nodes_5, x))
        y_interpolated_20.append(lagrange_interpolation(x_nodes_20, y_nodes_20, x))

    plot_lagrange_interpolation(y_interpolated_20, y_interpolated_5, x_values, y_values)

    y_interpolated_5 = [cubic_spline_interpolation(x_nodes_5, y_nodes_5, x) for x in x_values]
    y_interpolated_20 = [cubic_spline_interpolation(x_nodes_20, y_nodes_20, x) for x in x_values]
    plot_cubic_spline_interpolation(x_values, y_values, y_interpolated_5, y_interpolated_20)

    x_nodes_5 = x_values_ev[1::len(x_values_ev) // 5]
    y_nodes_5 = y_values_ev[1::len(y_values_ev) // 5]
    x_nodes_20 = x_values_ev[1::len(x_values_ev) // 10]
    y_nodes_20 = y_values_ev[1::len(y_values_ev) // 10]

    y_interpolated_5 = []
    y_interpolated_20 = []

    for x in x_values_ev:
        y_interpolated_5.append(lagrange_interpolation(x_nodes_5, y_nodes_5, x))
        y_interpolated_20.append(lagrange_interpolation(x_nodes_20, y_nodes_20, x))

    plot_lagrange_interpolation(y_interpolated_20, y_interpolated_5, x_values_ev, y_values_ev)

    y_interpolated_5 = [cubic_spline_interpolation(x_nodes_5, y_nodes_5, x) for x in x_values_ev]
    y_interpolated_20 = [cubic_spline_interpolation(x_nodes_20, y_nodes_20, x) for x in x_values_ev]
    plot_cubic_spline_interpolation(x_values_ev, y_values_ev, y_interpolated_5, y_interpolated_20)

if __name__ == "__main__":
    main()
