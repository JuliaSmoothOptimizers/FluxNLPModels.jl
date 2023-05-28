using CSV, DataFrames, Plots

# Read data from CSV file
df = CSV.read("sample\\csvFiles\\all_train_acc.csv", delim=',', DataFrame)

# Extract data columns
epochs = df[:, :Epoch]
accuracy_float16 = df[:, :Float16]
accuracy_float32 = df[:, :Float32]
accuracy_float32sr = df[:, :Float32SR]
accuracy_float64 = df[:, :Float64]

# Plotting
plot(epochs, accuracy_float16, label="Float16")
plot!(epochs, accuracy_float32, label="Float32")
plot!(epochs, accuracy_float32sr, label="Float32SR")
plot!(epochs, accuracy_float64, label="Float64")

xlabel!("Epoch")
title!("Accuracy vs. Epoch")
ylabel!("Accuracy")
plot!(legend=:outerbottom, legendcolumns=4)
# ylabel!("Loss")
# title!("Loss vs. Epoch")

# legend!(:topright)
# grid!(true)