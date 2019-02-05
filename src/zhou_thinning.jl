"""
```
zhou_thinning(binarized_img)
```

An algorithm for thinning a binarized image using both a flag map and an
inverted bitmap of the image. Will not work with non-binarized images

#Output
Returns an `AbstractArray` which is a binarised image with the thinned skeleton
of the original image. The skeleton is one-pixel thick and black

#Details

This algorithm uses a flag map and an inverted bitmap of the image. The flag map
is an integer array which contains only 0s and 1s, where 0 means the pixel in
the underlying image is flagged and 1 meaning not flagged. The inverted bitmap
is an integer array containing only 0s and 1s, where 0 represents a white pixel
in the original image and 1 represents a black pixel. Both the flag map and the
bitmap are the same size `n×m` as the image.

In the subsequent details, the following symbols are used to help describe
certain aspects of the algorithm:

- ``P_0`` is used to describe the pixel that the algorithm is currently assessing
- ``P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8`` are used to describe the 8 pixels
  surrounding ``P_0``. If ``P_0 = P_{[i][j]}`` (``i,j`` = row,column) then
  ``P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8`` correspond to ``P_{[i-1][j-1]},``
  ``P_{[i-1][j]},P_{[i-1][j+1]},P_{[i][j+1]},P_{[i+1][j+1]},P_{[i+1][j]},``
  ``P_{[i+1],[j-1]} \\; \\& \\; P_{[i][j-1]}`` respectively.
- ``Q_0,Q_1,Q_2,Q_3,Q_4,Q_5,Q_6,Q_7,Q_8`` represent positions on the flag map
  which correspond to ``P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8`` respectively.

Now we define following functions which are used in the algorithm:

```math
\\text{Previous Neighbourhood Funtion (PN):} \\quad
\\sum_{i=1}^{8} P_i
```
```math
\\text{Current Neighbourhood Funtion (CN):} \\quad
\\sum_{i=1}^{8} (P_i\\times Q_i)
```
```math
\\text{Transition Funtion (trans)} \\quad
\\sum_{i=1}^{8} (count(P_i))
\\quad \\text{where} \\quad
count(P_i) = \\begin{cases}
  1, & \\text{if } P_i \\times Q_i = 0 \\; \\text{and} \\; P_{i+1} \\times Q_{i+1} = 1\\\\
  0, & \\text{otherwise}
\\end{cases}
\\quad (\\text{note:} P_9 = P_1 \\text{ and } Q_9 = Q_1)
```

Certain conditions are used in the algorithm to flag pixels. They are as follows:
- condition 1: ``1 < CN(P) < 6``
- condition 2: ``trans(P) = 1``
- condition 3: The P and the 8 pixel neighbourhood matches 1 of 8 templates (see [1], section 2.2, Fig. 4 for more details)

Now for the steps of the algorithm:
1. For each pixel ``P`` in the bitmap
    (I) perform ``PN(P), CN(P), trans(P)``
    (II) check if the pixel bitmap value is 1  it satisfies condition 1 and (condition 2 or condition 3), flag it
2. Change the flagged pixels from 1 to 0 in the bitmap
3. Repeat until no pixels are flagged

See [1] for extra details and reasoning behind the algorithm

#Options

##Choices for `binarized_img`
Must be a binarised image where a single value is stored for each pixel (e.g.
of type `Gray{N0f8}`)

#Example
```julia
using ImageComponentAnalysis, ImageBinarization, TestImages

img = testimage("cameraman")
#required to binarize the image
img2 = binarize(AdaptiveThreshold(),img,15)
# Doesn't demonstrate algorithm capabilities, just how to use the function
zhou_thinning(img2)
```

#References
[1] Zhou, R., Quek, C. and Ng, G. (1995). A novel single-pass thinning algorithm and an effective set of performance criteria. Pattern Recognition Letters, [online] 16(12), pp.1267-1275. Available at: https://www.sciencedirect.com/science/article/pii/016786559500078X [Accessed 5 Feb. 2019].
"""
function zhou_thinning(binarized_img::AbstractArray)
    new_img = copy(binarized_img)
    zhou_thinning!(new_img)
    new_img
end

function zhou_thinning!(binarized_img::AbstractArray)
    neighbourhood = CartesianIndex.([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),
                                     (1,-1),(0,-1)])
    #create a border around img to avoid bounds errors
    bitmap = Int16.(padarray(binarized_img,Fill(1,(1,1),(1,1))))
    #flip_bit changes black to 1 and white to 0 as in the paper
    flip_bit(bitmap)
    smoothing_templates = [[0 1 0; 0 1 1; 0 0 0],[0 0 0; 0 1 1; 0 1 0],[0 0 0;
                            1 1 0; 0 1 0],[0 1 0; 1 1 0; 0 0 0]]
    flag_count = 1
    n_array = fill(1,3,3)
    while flag_count > 0
        flag_map = fill(1,size(bitmap))
        for i in CartesianIndices(binarized_img)
            if bitmap[i] == 1 && pn(i,bitmap,neighbourhood)
                cn_count = cn(i,bitmap,neighbourhood,flag_map)
                trans_count = trans(i,bitmap,neighbourhood,flag_map)
                if 1 < cn_count < 6
                    if trans_count == 1 || smoothing_check(i,bitmap,neighbourhood,smoothing_templates,n_array)
                        flag_map[i+CartesianIndex(1,1)] = 0
                    end
                end
            end
        end
        for i in CartesianIndices(binarized_img)
            if flag_map[i+CartesianIndex(1,1)] == 0
                bitmap[i] = 0
            end
        end
        flag_count = any(x->x==0,flag_map)
    end
    flip_bit(bitmap)
    for i in CartesianIndices(binarized_img)
        binarized_img[i] = bitmap[i]
    end
end

#Utility Functions
#Previous Neighbourhood function,c_pxl = current_pixel bmp = bitmap, n = neighbourhood
function pn(c_pxl::CartesianIndex, bmp::AbstractArray, n::Array)
    count = 0
    for i in 1:8
        count = count + bmp[c_pxl+n[i]]
    end
    count != 8 ? true : false
end

#Current Neighbourhood function, fm = flag_map
function cn(c_pxl::CartesianIndex, bmp::AbstractArray, n::Array, fm::AbstractArray)
    count = 0
    for i in 1:8
        count = count + bmp[c_pxl+n[i]]*fm[CartesianIndex(1,1)+c_pxl+n[i]]
    end
    count
end

#transition Function
function trans(c_pxl::CartesianIndex, bmp::AbstractArray, n::Array, fm::AbstractArray)
    count = 0
    for i in 1:8
        if bmp[c_pxl+n[i]]*fm[CartesianIndex(1,1)+c_pxl+n[i]] == 0
            if bmp[c_pxl+n[i%8+1]]*fm[CartesianIndex(1,1)+c_pxl+n[i%8+1]] == 1
                count = count + 1
            end
        end
    end
    count
end

# Checks the pixel neighbourhood to see if it matches any of the smoothing
# templates. Returns true if so, false otherwise. st = smoothing templates
function smoothing_check(c_pxl::CartesianIndex, bmp::AbstractArray, n::Array, st::AbstractArray, n_array::Array)
    for i in 1:8
        n_array[CartesianIndex(2,2)+n[i]] = bmp[c_pxl+n[i]]
    end
    for i in 1:4
        if n_array == st[i]
            return true
        end
    end
    false
end

function flip_bit(bmp::AbstractArray)
    for i in 1:length(bmp)
        bmp[i] == 0 ? bmp[i] = 1 : bmp[i] = 0
    end
end
