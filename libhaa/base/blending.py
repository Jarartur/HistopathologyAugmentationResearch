# # blending methods for artifact to wsi
# # TODO: rewrite

import warnings
import cv2
import numpy as np
import staintools
import scipy.ndimage as nd

def do_nothing(artifact, wsi_patch, artifact_mask, **kwargs):
    assert artifact.shape[:-1] == wsi_patch.shape[:-1] == artifact_mask.shape, f"{artifact.shape=} {wsi_patch.shape=} {artifact_mask.shape=}"
    artifact = artifact[..., :3] if artifact.shape[-1] == 4 else artifact
    return out

# def calculate_cdf(histogram):
#     """
#     This method calculates the cumulative distribution function
#     :param array histogram: The values of the histogram
#     :return: normalized_cdf: The normalized cumulative distribution function
#     :rtype: array
#     """
#     # Get the cumulative sum of the elements
#     cdf = histogram.cumsum()

#     # Normalize the cdf
#     normalized_cdf = cdf / float(cdf.max())

#     return normalized_cdf


# def calculate_lookup(src_cdf, ref_cdf):
#     """
#     This method creates the lookup table
#     :param array src_cdf: The cdf for the source image
#     :param array ref_cdf: The cdf for the reference image
#     :return: lookup_table: The lookup table
#     :rtype: array
#     """
#     lookup_table = np.zeros(256)
#     lookup_val = 0
#     for src_pixel_val in range(len(src_cdf)):
#         lookup_val = 0
#         for ref_pixel_val in range(len(ref_cdf)):
#             if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
#                 lookup_val = ref_pixel_val
#                 break
#         lookup_table[src_pixel_val] = lookup_val
#     return lookup_table


def bilateral_filter(out, mask, **kwargs):
    dtype = out.dtype
    bilateral_blur = cv2.bilateralFilter(
        out.astype("float32"),
        3,
        27,
        1,
    )
    mask_dilated = nd.binary_dilation(mask, iterations=3)
    # mask_dilated = np.repeat(mask_dilated[:,:,None], 3, 2)
    out = np.where(mask_dilated > 0, bilateral_blur, out)

    return out.astype(dtype)


# def center_matrix(transform, shape):
#     # A @ transform @ A^-1
#     center_x = (shape[1] - 1) / 2
#     center_y = (shape[0] - 1) / 2
#     m_translation = np.array(
#         [
#             [1, 0, center_x],
#             [0, 1, center_y],
#             [0, 0, 1],
#         ]
#     )
#     m_invert = np.array(
#         [
#             [1, 0, -center_x],
#             [0, 1, -center_y],
#             [0, 0, 1],
#         ]
#     )
#     centered_matrix = m_translation @ transform @ m_invert
#     return centered_matrix


def blend_output_dst(out, dst, mask):
    sigma = 9 #TODO make this dependend on the size of the mask
    mask = nd.gaussian_filter(mask.astype(float), sigma)
    if len(mask.shape) == 2:
        mask = np.repeat(mask[:, :, None], 3, 2)
    else:
        mask = mask
    out = mask * out + (1 - mask) * dst

    return out


# def smudge(src, dst, mask, **kwargs):
#     # TODO add color deformation before? (idea)

#     # # match_histograms(out, src, mask)
#     # out = cut_histograms(src, out, mask, **kwargs)

#     RCN = staintools.ReinhardColorNormalizer()
#     RCN.fit(src)
#     out = RCN.transform(dst)
#     out = nd.gaussian_filter(out, kwargs["sigma"])

#     out = blend_output_dst(out, dst, mask, kwargs["blending_sigma"])
#     # out = u.add_red_square(out)
#     return out


def focus_distortion(src, dst, mask):
    out = cv2.GaussianBlur(dst, [15, 15], 9)
    out = blend_output_dst(out, dst, mask)
    # out = u.add_red_square(out)
    return out


# def focus_distortion_transform(src, dst, mask, **kwargs):
#     out = cv2.GaussianBlur(dst, kwargs["kernel"], kwargs["sigma"])

#     aff_mat = np.eye(3)
#     aff_mat[0, 0] = kwargs["resize"]
#     aff_mat[1, 1] = kwargs["resize"]
#     aff_mat = center_matrix(aff_mat, mask.shape[::-1])
#     out = cv2.warpAffine(out, aff_mat[0:-1, :], mask.shape[::-1])

#     out = blend_output_dst(out, dst, mask, kwargs["blending_sigma"])
#     # out = u.add_red_square(out)
#     return out


def hard_clone(src, dst, mask):
    mask = mask[..., None]
    output = np.where(mask > 0, src, dst)

    output = bilateral_filter(output, mask)
    output = blend_output_dst(output, dst, mask)

    # output = u.add_red_square(output)
    return output


def cvClone(src, dst, mask):
    center = (
        np.round(0.5 * dst.shape[0]).astype(int),
        np.round(0.5 * dst.shape[1]).astype(int),
    )

    if len(mask.shape) == 2:
        mask_cv = mask[:, :, None] * 255
    else:
        mask_cv = mask * 255
    center = center[::-1]
    output = cv2.seamlessClone(src, dst, mask_cv, center, 2)
    output = blend_output_dst(output, dst, mask)
    # output = u.add_red_square(output)
    return output


def reinhard_staining(src, dst, mask):
    RCN = staintools.ReinhardColorNormalizer()
    RCN.fit(src)
    out = RCN.transform(dst)
    out = blend_output_dst(out, dst, mask)
    # out = u.add_red_square(out)
    return out


# def cut_histograms(src, dst, mask, **kwargs):
#     # Split the images into the different color channels
#     # b means blue, g means green and r means red
#     src_r, src_g, src_b = cv2.split(dst)

#     # Compute the b, g, and r histograms separately
#     # The flatten() Numpy method returns a copy of the array c
#     # collapsed into one dimension.
#     src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
#     src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
#     src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
#     ref_hist_blue, bin_3 = np.histogram(src_b.flatten(), 256, [0, 256])
#     ref_hist_green, bin_4 = np.histogram(src_g.flatten(), 256, [0, 256])
#     ref_hist_red, bin_5 = np.histogram(src_r.flatten(), 256, [0, 256])

#     if "cutoff" in kwargs:
#         ref_hist_blue[kwargs["cutoff"] :] = 0
#         ref_hist_green[kwargs["cutoff"] :] = 0
#         ref_hist_red[kwargs["cutoff"] :] = 0
#     if "cuton" in kwargs:
#         ref_hist_blue[: kwargs["cuton"]] = 0
#         ref_hist_green[: kwargs["cuton"]] = 0
#         ref_hist_red[: kwargs["cuton"]] = 0

#     # Compute the normalized cdf for the source and reference image
#     src_cdf_blue = calculate_cdf(src_hist_blue)
#     src_cdf_green = calculate_cdf(src_hist_green)
#     src_cdf_red = calculate_cdf(src_hist_red)
#     ref_cdf_blue = calculate_cdf(ref_hist_blue)
#     ref_cdf_green = calculate_cdf(ref_hist_green)
#     ref_cdf_red = calculate_cdf(ref_hist_red)

#     # Make a separate lookup table for each color
#     blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
#     green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
#     red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

#     # Use the lookup function to transform the colors of the original
#     # source image
#     blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
#     green_after_transform = cv2.LUT(src_g, green_lookup_table)
#     red_after_transform = cv2.LUT(src_r, red_lookup_table)

#     # Put the image back together
#     image_after_matching = cv2.merge(
#         [red_after_transform, green_after_transform, blue_after_transform]
#     )
#     image_after_matching = cv2.convertScaleAbs(image_after_matching)

#     return image_after_matching


# def match_histograms(src, dst, mask, **kwargs):
#     """
#     This method matches the source image histogram to the
#     reference signal
#     :param image src_image: The original source image
#     :param image  ref_image: The reference image
#     :return: image_after_matching
#     :rtype: image (array)
#     """
#     # Split the images into the different color channels
#     # b means blue, g means green and r means red
#     src_r, src_g, src_b = cv2.split(dst)
#     ref_r, ref_g, ref_b = cv2.split(src)

#     # Compute the b, g, and r histograms separately
#     # The flatten() Numpy method returns a copy of the array c
#     # collapsed into one dimension.
#     src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
#     src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
#     src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
#     ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0, 256])
#     ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0, 256])
#     ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0, 256])

#     if "cutoff" in kwargs:
#         ref_hist_blue[kwargs["cutoff"] :] = 0
#         ref_hist_green[kwargs["cutoff"] :] = 0
#         ref_hist_red[kwargs["cutoff"] :] = 0
#     if "cuton" in kwargs:
#         ref_hist_blue[: kwargs["cuton"]] = 0
#         ref_hist_green[: kwargs["cuton"]] = 0
#         ref_hist_red[: kwargs["cuton"]] = 0

#     # Compute the normalized cdf for the source and reference image
#     src_cdf_blue = calculate_cdf(src_hist_blue)
#     src_cdf_green = calculate_cdf(src_hist_green)
#     src_cdf_red = calculate_cdf(src_hist_red)
#     ref_cdf_blue = calculate_cdf(ref_hist_blue)
#     ref_cdf_green = calculate_cdf(ref_hist_green)
#     ref_cdf_red = calculate_cdf(ref_hist_red)

#     # Make a separate lookup table for each color
#     blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
#     green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
#     red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

#     # Use the lookup function to transform the colors of the original
#     # source image
#     blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
#     green_after_transform = cv2.LUT(src_g, green_lookup_table)
#     red_after_transform = cv2.LUT(src_r, red_lookup_table)

#     # Put the image back together
#     image_after_matching = cv2.merge(
#         [red_after_transform, green_after_transform, blue_after_transform]
#     )
#     image_after_matching = cv2.convertScaleAbs(image_after_matching)

#     return image_after_matching
