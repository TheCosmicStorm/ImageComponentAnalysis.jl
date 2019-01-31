module ImageComponentAnalysis

using Images

abstract type ComponentAnalysisAlgorithm end
struct ContourTracing <: ComponentAnalysisAlgorithm end

include("contour_tracing.jl")
include("zhou_thinning.jl")

export
    label_components,
	ContourTracing,
	zhou_thinning
end # module
