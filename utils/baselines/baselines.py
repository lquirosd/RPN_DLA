import sys
import cv2
import numpy as np
import argparse
import glob
from tqdm import tqdm
from xmlPAGE import pageData
import polyapprox as pa


def basic_baseline(Oimg, Lpoly, args):
    """
    """
    # --- Oimg = image to find the line
    # --- Lpoly polygon where the line is expected to be
    try:
        minX = Lpoly[:, 0].min()
        maxX = Lpoly[:, 0].max()
        minY = Lpoly[:, 1].min()
        maxY = Lpoly[:, 1].max()
    except:
        print(Lpoly)
        return(False, None)
    mask = np.zeros(Oimg.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, Lpoly, (255, 255, 255))
    res = cv2.bitwise_and(Oimg, mask)
    bRes = Oimg[minY:maxY, minX:maxX]
    bMsk = mask[minY:maxY, minX:maxX]
    try:
        bRes = cv2.cvtColor(bRes, cv2.COLOR_RGB2GRAY)
        _, bImg = cv2.threshold(bRes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, cols = bImg.shape
    except:
        print(minY,maxY,minX,maxX)
        return(False, None)
    # --- remove black halo around the image
    bImg[bMsk[:, :, 0] == 0] = 255
    Cs = np.cumsum(abs(bImg - 255), axis=0)
    maxPoints = np.argmax(Cs, axis=0)
    Lmsk = np.zeros(bImg.shape)
    points = np.zeros((cols, 2), dtype="int")
    # --- gen a 2D list of points
    for i, j in enumerate(maxPoints):
    	points[i, :] = [i, j]
    # --- remove points at post 0, those are very probable to be blank columns
    points2D = points[points[:, 1] > 0]
    if points2D.size <= 15:
    # --- there is no real line
    	return (False, [[0, 0]])
    if args.approx_alg == "optimal":
    # --- take only 100 points to build the baseline
    	if points2D.shape[0] > args.max_vertex:
        	points2D = points2D[
        	np.linspace(
            	0, points2D.shape[0] - 1, args.max_vertex, dtype=np.int
        	)
        	]
    	(approxError, approxLin) = pa.poly_approx(
        	points2D, args.num_segments, pa.one_axis_delta
    	)
    elif args.approx_alg == "trace":
    	approxLin = pa.norm_trace(points2D, args.num_segments)
    else:
    	approxLin = points2D
    approxLin[:, 0] = approxLin[:, 0] + minX
    approxLin[:, 1] = approxLin[:, 1] + minY
    return (True, approxLin)


def main():
    ALGORITHMS = {"basic":basic_baseline,}
    parser = argparse.ArgumentParser(
        description="This script support some basic baseline related extractions"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="path to images files directory.",
    )
    parser.add_argument(
        "--page_dir",
        type=str,
        default=None,
        help="path to page-xml files directory.",
    )
    parser.add_argument(
        "--page_ext",
        type=str,
        default="xml",
        help='Extensiong of PAGE-XML files. Default "xml"',
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="basic",
        help="Algorithm to be used to gen the baselines",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Path to save new generated xml files",
    )
    parser.add_argument(
        "--must_line",
        type=str,
        nargs="+",
        default="",
        help="Search for baselines on this regions even if no TextLine found",
    )

    args = parser.parse_args()
    args.num_segments = 4
    args.max_vertex = 30
    args.approx_alg = "optimal"
    get_baselines = ALGORITHMS[args.algorithm]
    files = glob.glob(args.page_dir+"/*."+args.page_ext)
    for page in tqdm(files):
        data = pageData(page)
        data.parse()
        file_name = data.get_image_path()
        if file_name is None:
            file_name = args.img_dir + '/' + data.get_image_name()
        img = cv2.imread(file_name)
        lines = data.get_region("TextLine")
        if lines is None:
            print("INFO: No TextLines found on page {}".format(page))
            data.save_xml(f_path=args.out_dir + '/'+ data.full_name)
            continue
        for line in lines:
            coords = data.get_coords(line)
            (valid, baseline) = get_baselines(img, coords, args)
            if valid == True:
                data.add_baseline(pa.points_to_str(baseline), line)
            else:
                print("No baseline found for line {}".format(data.get_id(line)))
        #--- check for TextRegion without TextLine
        regions = data.get_region("TextRegion")
        idx = 0
        for region in regions:
            r_type = data.get_region_type(region)
            if r_type in args.must_line:
                lines = data.get_childs("TextLine", parent=region)
                if lines == None:
                    coords = data.get_coords(region)
                    (valid, baseline) = get_baselines(img, coords, args)
                    if valid == True:
                        line = data.add_element("TextLine", "TextLine_extra" + str(idx), r_type, pa.points_to_str(coords), parent=region)
                        idx += 1
                        data.add_baseline(pa.points_to_str(baseline), line)
        data.save_xml(f_path=args.out_dir + '/'+ data.full_name)







if __name__== "__main__":
    main()
