import numpy as np

def math_median(stack, number, tolerance):
    return np.median(stack[-number:])

def math_poly(degree, stack, number, tolerance):
	stack = stack[-number:]
	t = np.array([i for i in xrange(number)])
	z = np.polyfit(t, stack, degree)
	p = np.poly1d(z)
	x = p(number)
	temp = math_median(stack, number, tolerance)
	if np.abs(x-temp)>tolerance*np.max(np.abs(stack-temp)):return temp
	return x


def predict(stack, tolerance = 2):
    #print repr(stack)
    number = min(4, len(stack))
	# switch case
    switcher = {
        0: lambda a,b,c: None,
        1: lambda a,b,c: stack[0],
        2: math_median,
        3: math_median,
        4: math_median,
        5: lambda a,b,c: math_poly(2, a,b,c),
        6: lambda a,b,c: math_poly(2, a,b,c)}
    temp = switcher.get(number, lambda a,b,c: math_poly(3, a,b,c))(stack, number, tolerance)
    if temp is None:return None
    return (temp + math_median(stack, 2, tolerance))/2


def frame_prediction(frames, element, tolerance=2):
    assert element in ["centre_x", "centre_y", "radius", "magic_wall", "angle"]
    if len(frames) == 0: return None

    if element == "centre_x":
        return predict(np.array([i.centre_x for i in frames]), tolerance)
    if element == "centre_y":
        return predict(np.array([i.centre_y for i in frames]), tolerance)
    if element == "radius":
        return predict(np.array([i.radius for i in frames]), tolerance)
    if element == "magic_wall":
        return predict(np.array([i.intercept for i in frames]), tolerance)
    if element == "angle":
        return predict(np.array([i.slope for i in frames]), tolerance)
    
    
"""
def predict_test():
	a = np.array([ 296.5,  297.5,  299.5,  300.5,  298.5,  300.5,  299.5,  303.5,
        299.5,  298.5,  293.5,  296.5,  294.5,  293.5,  288.5,  287.5,
        283.5,  281.5,  279.5,  276.5,  275.5,  273.5,  275.5,  276.5,
        279.5,  278.5,  289.5,  287.5,  288.5,  279.5,  274.5,  270.5,
        274.5,  273.5,  270.5,  264.5,  263.5,  263.5,  260.5,  259.5,
        258.5,  260.5,  259.5,  263.5,  260.5,  261.5,  264.5,  262.5,
        264.5,  265.5,  262.5,  256.5,  258.5,  261.5,  259.5,  256.5,
        258.5,  257.5,  260.5,  259.5,  258.5,  261.5,  261.5,  260.5,
        257.5,  256.5,  256.5,  261.5,  261.5,  261.5,  260.5,  260.5,
        264.5,  261.5,  266.5,  261.5,  264.5,  265.5,  263.5,  263.5,
        263.5,  261.5,  260.5,  262.5,  260.5,  259.5,  258.5,  258.5,
        262.5,  261.5,  261.5,  260.5,  262.5,  262.5,  262.5,  261.5,
        262.5,  260.5,  258.5,  264.5,  260.5,  266.5,  260.5,  261.5,
        263.5,  261.5,  260.5,  262.5,  261.5,  259.5,  264.5,  260.5,
        263.5,  260.5,  260.5,  262.5,  258.5,  259.5,  261.5,  258.5,
        257.5,  258.5,  257.5,  257.5,  259.5,  256.5,  257.5,  259.5,
        259.5,  258.5,  259.5,  259.5,  262.5,  260.5,  259.5,  259.5,
        261.5,  258.5,  259.5,  261.5,  262.5,  260.5,  260.5,  257.5,
        257.5,  257.5,  264.5,  261.5,  260.5,  260.5,  261.5,  258.5,
        260.5,  264.5,  268.5,  270.5,  270.5,])
	guess = np.array([predict(a[:i]) for i in xrange(a.size)])

	print "actual\t\tpredicted"
	for i,j in zip(a, guess):
		print i, "\t\t" , j

predict_test()
"""