using Documenter, FluxNLPModels

makedocs(
  modules = [FluxNLPModels],
  doctest = true,
  # linkcheck = true,
  format = Documenter.HTML(
    assets = ["assets/style.css"],
    prettyurls = get(ENV, "CI", nothing) == "true",
  ),
  sitename = "FluxNLPModels.jl",
  pages = ["Home" => "index.md", "Tutorial" => "tutorial.md", "Reference" => "reference.md"],
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/FluxNLPModels.jl.git",
  push_preview = true,
  devbranch = "main",
)
