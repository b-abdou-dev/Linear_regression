# implementing linear regression line equation from:
#   https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/regression-analysis/find-a-linear-regression-equation/

from matplotlib import pyplot as plt


def find_a_and_b(n, x_list, y_list):
    sum_xs = sum(x_list)
    sum_ys = sum(y_list)
    sum_xs_square = sum(x_list) ** 2
    sum_x_squares = sum([x * x for x in x_list])
    sum_x_times_y = sum(x * y for x, y in zip(x_list, y_list))

    a = (sum_ys * sum_x_squares - sum_xs * sum_x_times_y) / (n * sum_x_squares - sum_xs_square)
    b = (n * sum_x_times_y - sum_xs * sum_ys) / (n * sum_x_squares - sum_xs_square)

    return a,b





if __name__ == '__main__':

    n = 6     # sample size
    x_list = [43, 21, 25, 42, 57, 59]
    y_list = [99, 65, 79, 75, 87, 81]
    x_list.sort()
    y_list.sort()

    a,b = find_a_and_b(n, x_list, y_list)

    print("value of a is", a, "value of b is", b)

    # find the values of ys based on given xs
    xs_values = [1, 100]
    ys_values = [0, 0]
    for i in range(0, 2):
        ys_values[i] = b * xs_values[i] + a

    # estimate value of y given value of x
    x_given_value = 80
    estimated_y_value = b * x_given_value + a


 # plotting the points

    plt.plot(x_list, y_list, marker="o", label="Given Data")
    plt.plot(xs_values, ys_values, label="Linear Regression")
    plt.plot(x_given_value, estimated_y_value, marker="x", label="estimated")

    # naming the x axis
    plt.xlabel('x - vaules')
    # naming the y axis
    plt.ylabel('y - values')

    # giving a title to my graph
    plt.title('Linear Regression Plot')

    # show a legend on the plot
    plt.legend()

    # save the plot
    plt.savefig('plot.png')

    # function to show the plot
    plt.show()


