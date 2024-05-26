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

if __name__ == "__main__":
    main()
