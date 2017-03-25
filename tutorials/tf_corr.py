
def corr(y_true, y_pred)
    N = y_true.get_shape[0]
    mean1, var1 = tf.nn.moments(y_true,axes=[1])
    mean2, var2 = tf.nn.moments(y_pred,axes=[1])
    num = -1*(tf.reduce_sum(y_true*y_pred) - N*mean1*mean2) 
    denom =(N*tf.sqrt(var1)*tf.sqrt(var2))


X= np.random.randn(20,4)
print np.corrcoef(x.T)


def cca_loss(x,y):
    """
    Say x = (20,3) , y = (20,3)
    """
    xt = tf.transpose(x) # (3,20)
    yt = tf.transpose(y) # (3,20)
    sxx = tf.matmul(xt,x) # (3,3)
    syy = tf.matmul(yt,y) # (3,3)
    xxy = tf.matmul(x,yt) # (20,3) * (3,20) = (20,20)
    num = tf.matmul(tf.matmul(xt,sxy),y) # (3,20) * (20,20)*(20,3)
    denom1 = tf.sqrt(tf.matmul(x))



def corrcoef(X):
    X = X - np.mean(X,axis=0) # zero-mean each feature (column)
    #c = cov(x, y, rowvar)
    c = np.dot(X.T,X)
    try:
        d = np.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    
    stddev = np.sqrt(d.real)

    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c