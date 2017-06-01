centers = [(0.4, 0.4), (-0.4, -0.4)]
width = 0.3
def field_fun(x, y, centers, ls):
    ret = 1.
    for c in centers:
        ret = ret + np.exp(-((x - c[0])**2 + (y - c[1])**2) / ls**2)
    return ret

field_fun_x = autograd.elementwise_grad(field_fun)
field_fun_y = autograd.elementwise_grad(field_fun, 1)