# test example taken from Flux quickstart guide ([https://fluxml.ai/Flux.jl/stable/models/quickstart/](https://fluxml.ai/Flux.jl/stable/models/quickstart/))

noisy = rand(Float32, 2, 1000)                                  
truth = [xor(col[1]>0.5, col[2]>0.5) for col in eachcol(noisy)]
target = Flux.onehotbatch(truth, [true, false])

model16_cpu = Chain(
    Dense(2 => 3, tanh),   # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2),
    softmax) |> f16
model32_cpu = Chain(
    Dense(2 => 3, tanh),   # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2),
    softmax) |> f32
model32_gpu = Chain(
    Dense(2 => 3, tanh),   # activation function inside layer
    BatchNorm(3),
    Dense(3 => 2),
    softmax) |> gpu
loader_cpu = Flux.DataLoader((noisy, target) |> f16, batchsize=64);
loader_gpu = Flux.DataLoader((noisy, target) |> gpu, batchsize=64);
  
@testset "FluxNLPModel ill-instanciation checks" begin
  try 
    FluxNLPModel([model16_cpu,model32_gpu],loader_cpu,loader_cpu)
    @test false
  catch
    @test true
  end
  try 
    FluxNLPModel([model16_cpu,model32_cpu],loader_gpu,loader_cpu)
    @test false
  catch
    @test true
  end
  try 
    FluxNLPModel([model16_cpu,model32_cpu],loader_cpu,loader_gpu)
    @test false
  catch
    @test true
  end
  try 
    FluxNLPModel([model32_cpu,model16_cpu],loader_cpu,loader_cpu) # wrong model order
    @test false
  catch
    @test true
  end
end

@testset "obj/grad FP formats consistency" begin
  nlp = FluxNLPModel([model16_cpu,model32_cpu],loader_cpu,loader_cpu)
  x16,_ = Flux.destructure(model16_cpu)
  @test typeof(obj(nlp,x16)) == eltype(x16)
  @test eltype(grad(nlp,x16)) == eltype(x16)
  g16 = similar(x16)
  o16, g16 = objgrad!(nlp,x16,g16)
  @test typeof(o16) == eltype(x16)
  @test eltype(g16) == eltype(x16)
  x32 = Float32.(x16)
  @test typeof(obj(nlp,x32)) == eltype(x32)
  @test eltype(grad(nlp,x32)) == eltype(x32)
  g32 = similar(x32)
  o32, g32 = objgrad!(nlp,x32,g32)
  @test typeof(o32) == eltype(x32)
  @test eltype(g32) == eltype(x32)

  # gpu
  nlp = FluxNLPModel([model32_gpu],loader_gpu,loader_gpu)
  x32,_ = Flux.destructure(model32_gpu)
  @test typeof(obj(nlp,x32)) == eltype(x32)
  @test eltype(grad(nlp,x32)) == eltype(x32)
  g32 = similar(x32)
  o32, g32 = objgrad!(nlp,x32,g32)
  @test typeof(o32) == eltype(x32)
  @test eltype(g32) == eltype(x32)
end