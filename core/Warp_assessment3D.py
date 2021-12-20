import numpy as np


def warp_assessment3D(u_GT, u_est, OF_index):

    if len(u_GT) == 3:
        uest_hold = np.concatenate((u_est[0][OF_index].flatten()[:, np.newaxis], u_est[1][OF_index].flatten()[:, np.newaxis], u_est[2][OF_index].flatten()[:, np.newaxis]), axis=1)
        ug_hold = np.concatenate((u_GT[0][OF_index].flatten()[:, np.newaxis], u_GT[1][OF_index].flatten()[:, np.newaxis], u_GT[2][OF_index].flatten()[:, np.newaxis]), axis=1)
    else:
        uest_hold = np.concatenate((u_est[0][OF_index].flatten()[:, np.newaxis], u_est[1][OF_index].flatten()[:, np.newaxis]), axis=0)
        ug_hold = np.concatenate((u_GT[0][OF_index].flatten()[:, np.newaxis], u_GT[1][OF_index].flatten()[:, np.newaxis]), axis=0)

    Error_Data = {}
    # ** ** ** *Absolute Error Analysis ** ** ** *
    # (also known as Endpoint Error)
    # Calculate the error:
    Error_Data['Abs_Error'] = np.sqrt(np.sum(np.power(np.abs(uest_hold - ug_hold), 2), axis=1))

    # Median Error:
    Error_Data['Abs_Error_median'] = np.median(Error_Data['Abs_Error'], axis=0)

    # Mean Error:
    Error_Data['Abs_Error_mean'] = np.mean(Error_Data['Abs_Error'], axis=0)

    # ** ** ** *Angular Error Analysis ** ** ** *
    # Calculate the error:
    Error_Data['Angle_Error'] = np.real(np.arccos((1 + np.sum(uest_hold * ug_hold, axis=1)) / (np.sqrt(1 + np.sum(np.power(uest_hold, 2), axis=1)) * np.sqrt(1 + np.sum(np.power(ug_hold, 2), axis=1))))) * 180 / np.pi
    OF_index_angle = ~np.isnan(Error_Data['Angle_Error'])
    # Median Error:
    Error_Data['Angle_Error_median'] = np.median(Error_Data['Angle_Error'][OF_index_angle], axis=0)

    # Mean Error:
    Error_Data['Angle_Error_Mean'] = np.mean(Error_Data['Angle_Error'][OF_index_angle], axis=0)

    return Error_Data


if __name__ == '__main__':

    u_GT = (u_GTx, u_GTy)  # tuple
    u_est = (u_estx, u_esty)  # tuple
    OF_index = u_GT[0] != np.nan  # *  u_GT[0] >= 0
    warp_assessment3D(u_GT, u_est, OF_index)