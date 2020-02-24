import numpy as np



def non_max_suppression_by_grid(boxes, grid_size_scale: float=1):
    '''
    performs non maxima suppression based elements clustered in a grid
    :param boxes: x1,y1,x2,y1,label,prob
    :param grid_size_scale: if 1 grid size equal mean object size 
    :return: filtered list
    '''

    # calculate mean object size
    grid_size_x = np.mean(boxes[:, 2] - boxes[:, 0]) * grid_size_scale
    grid_size_y = np.mean(boxes[:, 3] - boxes[:, 1]) * grid_size_scale

    # calculate the number of grid cells needed
    num_cells_x = np.ceil(boxes[:, 2].max() / grid_size_x).astype(np.uint)
    num_cells_y = np.ceil(boxes[:, 3].max() / grid_size_y).astype(np.uint)

    # calculate the center of each object
    center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
    center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2

    # calculate the center indexes
    index_x = np.floor(center_x / grid_size_x).astype(np.uint)
    index_y = np.floor(center_y / grid_size_y).astype(np.uint)

    # calculate a global index for each cell
    grid_indexes = index_x * num_cells_x + index_y



    # Sort grid index by prob
    grid_indexes_sorted = []
    for i in list(set(grid_indexes)):
        grid_indexes_sorted.append((i, max(boxes[grid_indexes == i][:, 5])))



    # iterate list of unique indexes
    index_to_keep = []
    for i in list(set(grid_indexes)):
        nonzero_index = np.nonzero(grid_indexes == i)[0]
        # filter by minimum distance

        grid_cell_result = {}
        # find for the current grid position for each label
        # the cell with the highest probability
        for element in nonzero_index:
            label = boxes[element][4]
            prob = boxes[element][5]
            if label not in grid_cell_result:
                grid_cell_result[label] = {'Prob': prob, 'Index': element}
            elif grid_cell_result[label]['Prob'] < prob:
                grid_cell_result[label]['Prob'] = prob

        for key in grid_cell_result:
            index_to_keep.append(grid_cell_result[key]['Index'])

    return boxes[index_to_keep]

