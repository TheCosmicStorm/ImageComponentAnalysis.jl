"""
```
zhou_thinning(binarized_img)
```


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
