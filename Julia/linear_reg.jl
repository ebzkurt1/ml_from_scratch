using DataFrames
using Plots
using MLDatasets: BostonHousing, Iris


"""Loading datasets for the given tasks"""
function load_dataset(ml_task="regression")
    if ml_task == "regression"
        return BostonHousing.features()', BostonHousing.targets()'
    elseif ml_task == "classification"
        return Iris.features(), Iris.labels()
    end
end


"""Making linear regression prediction using data and weights"""
function linear_reg_pred(X, w)
    return X * w
end


"""MSE loss function"""
function mse_loss(X, y, w)
    y_hat = linear_reg_pred(X, w)
    return sum(y - y_hat) / size(y)[1]
end


"""Linear regression algortihm"""
function linear_reg(X, Y)
    X_with_bias = hcat(ones(size(X)[1],1),X)
    w = inv(X_with_bias' * X_with_bias) * (X_with_bias' * Y)
    mse = mse_loss(X_with_bias, y, w)
    println("Linear Regression MSE error : $mse")
    return linear_reg_pred(X_with_bias, w), w
end

gr()
X, y = load_dataset();
y_hat, weights = linear_reg(X, y);
plot(collect(1:size(y)[1]), y, seriestype = :scatter, title = "Linear Regression Result");
plot!(collect(1:size(y)[1]), y_hat, seriestype = :scatter)
