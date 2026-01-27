import concurrent.futures
import fnmatch
import json
import sys

import hist
import numpy as np
import uproot
from spritz.framework.framework import (
    add_dict_iterable,
    get_analysis_dict,
    get_fw_path,
    read_chunks,
)

path_fw = get_fw_path()


def renorm(h, xs, sumw, lumi):
    scale = xs * 1000 * lumi / sumw
    # print(scale)
    _h = h.copy()
    a = _h.view(True)
    a.value = a.value * scale
    a.variance = a.variance * scale * scale
    return _h


def hist_move_content(h, ifrom, ito):
    """
    Moves content of a histogram from `ifrom` bin to `ito` bin.
    Content and sumw2 of bin `ito` will be the sum of the original `ibin`
    and `ito`.
    Content and sumw2 of bin `ifrom` will be 0.
    Modifies in place the histogram.

    Parameters
    ----------
    h : hist
        Histogram
    ifrom : int
        the index of the bin where content will be reset
    ito : int
        the index of the bin where content will be the sum
    """
    dimension = len(h.axes)
    # numpy view is a numpy array containing two keys, value
    # and variances for each bin
    numpy_view = h.view(True)
    content = numpy_view.value
    sumw2 = numpy_view.variance

    if dimension == 1:
        content[ito] += content[ifrom]
        content[ifrom] = 0.0

        sumw2[ito] += sumw2[ifrom]
        sumw2[ifrom] = 0.0

    elif dimension == 2:
        content[ito, :] += content[ifrom, :]
        content[ifrom, :] = 0.0
        content[:, ito] += content[:, ifrom]
        content[:, ifrom] = 0.0

        sumw2[ito, :] += sumw2[ifrom, :]
        sumw2[ifrom, :] = 0.0
        sumw2[:, ito] += sumw2[:, ifrom]
        sumw2[:, ifrom] = 0.0

    elif dimension == 3:
        content[ito, :, :] += content[ifrom, :, :]
        content[ifrom, :, :] = 0.0
        content[:, ito, :] += content[:, ifrom, :]
        content[:, ifrom, :] = 0.0
        content[:, :, ito] += content[:, :, ifrom]
        content[:, :, ifrom] = 0.0

        sumw2[ito, :, :] += sumw2[ifrom, :, :]
        sumw2[ifrom, :, :] = 0.0
        sumw2[:, ito, :] += sumw2[:, ifrom, :]
        sumw2[:, ifrom, :] = 0.0
        sumw2[:, :, ito] += sumw2[:, :, ifrom]
        sumw2[:, :, ifrom] = 0.0


def hist_fold(h, fold_method) -> None:
    """
    Fold a histogram (hist object)

    Parameters
    ----------
    h : hist
        Histogram to fold, will be modified in place (aka no copy)
    fold_method : int
        choices 0: no fold
        choices 1: fold underflow
        choices 2: fold overflow
        choices 3: fold both underflow and overflow
    """
    if fold_method == 1 or fold_method == 3:
        hist_move_content(h, 0, 1)
    if fold_method == 2 or fold_method == 3:
        hist_move_content(h, -1, -2)


def hist_unroll(h) -> hist.Hist:
    """
    Unrolls n-dimensional histogram

    Parameters
    ----------
    h : hist
        Histogram to unroll

    Returns
    -------
    hist
        Unrolled 1-dimensional histogram
    """
    dimension = len(h.axes)
    if dimension != 2:
        raise Exception(
            "Error in hist_unroll: can only unroll 2D histograms, while got ",
            dimension,
            "dimensions",
        )

    numpy_view = h.view()  # no under/overflow!
    nx = numpy_view.shape[0]
    ny = numpy_view.shape[1]
    h_unroll = hist.Hist(hist.axis.Regular(nx * ny, 0, nx * ny), hist.storage.Weight())

    numpy_view_unroll = h_unroll.view()
    numpy_view_unroll.value = numpy_view.value.T.flatten()
    numpy_view_unroll.variance = numpy_view.variance.T.flatten()

    return h_unroll


def make_new_histo(h):
    # numpy_view = h.view()  # no under/overflow!
    # h_unroll = hist.Hist(hist.axis.Regular(10, 0, 10), hist.storage.Weight())

    # numpy_view_unroll = h_unroll.view()
    # numpy_view_unroll.value = numpy_view.value[15:]
    # numpy_view_unroll.variance = numpy_view.variance[15:]

    # return h_unroll
    return h


def get_variations(h):
    axis = h.axes[-1]
    variation_names = [axis.value(i) for i in range(len(axis.centers))]
    return variation_names


def blind(region, variable, edges):
    if "sr" in region and "dnn" in variable:
        return np.arange(0, len(edges)) > len(edges) / 2


lumi_to_year = {
    19: "2016pre",
    16: "2016post",
    41: "2017",
    59: "2018",
}

corrections = {
    "2016pre": {
        0: {
            "nom": np.array(
                [0.84813563, 0.83277124, 0.81740684, 0.80204245, 0.78667805]
            ),
            "up": np.array(
                [0.85462306, 0.84283736, 0.83105165, 0.81926595, 0.80748024]
            ),
            "down": np.array(
                [0.8416482, 0.82270511, 0.80376203, 0.78481895, 0.76587586]
            ),
        },
        1: {
            "nom": np.array(
                [0.88953666, 0.87080455, 0.85207244, 0.83334033, 0.81460821]
            ),
            "up": np.array([0.91152348, 0.9052599, 0.89899632, 0.89273274, 0.88646917]),
            "down": np.array(
                [0.86754984, 0.8363492, 0.80514855, 0.77394791, 0.74274726]
            ),
        },
        2: {
            "nom": np.array(
                [0.95081365, 0.92744587, 0.90407808, 0.8807103, 0.85734251]
            ),
            "up": np.array([0.96382017, 0.9478179, 0.93181564, 0.91581337, 0.89981111]),
            "down": np.array(
                [0.93780713, 0.90707383, 0.87634052, 0.84560722, 0.81487392]
            ),
        },
        3: {
            "nom": np.array([0.98392791, 0.94701951, 0.91011111, 0.8732027, 0.8362943]),
            "up": np.array(
                [1.00206265, 0.97524096, 0.94841927, 0.92159758, 0.89477589]
            ),
            "down": np.array(
                [0.96579317, 0.91879806, 0.87180294, 0.82480783, 0.77781271]
            ),
        },
        4: {
            "nom": np.array(
                [1.04060586, 1.0062315, 0.97185713, 0.93748276, 0.90310839]
            ),
            "up": np.array(
                [1.08356267, 1.07401103, 1.06445938, 1.05490774, 1.04535609]
            ),
            "down": np.array(
                [0.99764906, 0.93845197, 0.87925488, 0.82005778, 0.76086069]
            ),
        },
    },
    "2016post": {
        0: {
            "nom": np.array(
                [0.87440537, 0.85133157, 0.82825777, 0.80518397, 0.78211016]
            ),
            "up": np.array([0.89072004, 0.8766657, 0.86261136, 0.84855702, 0.83450268]),
            "down": np.array(
                [0.8580907, 0.82599744, 0.79390417, 0.76181091, 0.72971764]
            ),
        },
        1: {
            "nom": np.array(
                [0.92807156, 0.90824329, 0.88841501, 0.86858673, 0.84875846]
            ),
            "up": np.array(
                [0.94123728, 0.92889529, 0.91655329, 0.90421129, 0.89186929]
            ),
            "down": np.array(
                [0.91490584, 0.88759129, 0.86027673, 0.83296218, 0.80564762]
            ),
        },
        2: {
            "nom": np.array(
                [0.99366151, 0.96040688, 0.92715225, 0.89389762, 0.86064299]
            ),
            "up": np.array(
                [1.03214802, 1.02069775, 1.00924748, 0.99779721, 0.98634694]
            ),
            "down": np.array(
                [0.95517499, 0.900116, 0.84505701, 0.78999803, 0.73493904]
            ),
        },
        3: {
            "nom": np.array(
                [1.04203917, 1.01542667, 0.98881418, 0.96220169, 0.9355892]
            ),
            "up": np.array([1.09001917, 1.09008415, 1.09014913, 1.09021412, 1.0902791]),
            "down": np.array(
                [0.99405916, 0.9407692, 0.88747923, 0.83418926, 0.7808993]
            ),
        },
        4: {
            "nom": np.array(
                [1.09615696, 1.00194672, 0.90773648, 0.81352624, 0.71931601]
            ),
            "up": np.array([1.16377524, 1.10931499, 1.05485475, 1.0003945, 0.94593426]),
            "down": np.array(
                [1.02853867, 0.89457844, 0.76061821, 0.62665798, 0.49269776]
            ),
        },
    },
    "2017": {
        0: {
            "nom": np.array(
                [0.88703308, 0.87207826, 0.85712343, 0.8421686, 0.82721378]
            ),
            "up": np.array(
                [0.90457948, 0.89929675, 0.89401402, 0.88873128, 0.88344855]
            ),
            "down": np.array(
                [0.86948668, 0.84485976, 0.82023284, 0.79560592, 0.770979]
            ),
        },
        1: {
            "nom": np.array(
                [0.92690005, 0.91361323, 0.90032641, 0.88703959, 0.87375277]
            ),
            "up": np.array([0.93353423, 0.924025, 0.91451578, 0.90500655, 0.89549733]),
            "down": np.array(
                [0.92026587, 0.90320146, 0.88613704, 0.86907262, 0.85200821]
            ),
        },
        2: {
            "nom": np.array(
                [0.99012571, 0.96838933, 0.94665295, 0.92491657, 0.90318019]
            ),
            "up": np.array([1.0109195, 1.00095781, 0.99099611, 0.98103442, 0.97107273]),
            "down": np.array(
                [0.96933191, 0.93582085, 0.90230978, 0.86879872, 0.83528765]
            ),
        },
        3: {
            "nom": np.array(
                [1.03138708, 0.99572057, 0.96005406, 0.92438754, 0.88872103]
            ),
            "up": np.array(
                [1.08176822, 1.07425463, 1.06674105, 1.05922746, 1.05171388]
            ),
            "down": np.array(
                [0.98100595, 0.91718651, 0.85336706, 0.78954762, 0.72572818]
            ),
        },
        4: {
            "nom": np.array(
                [1.06586984, 1.02145914, 0.97704844, 0.93263774, 0.88822704]
            ),
            "up": np.array([1.10273239, 1.07983656, 1.05694073, 1.0340449, 1.01114907]),
            "down": np.array(
                [1.02900729, 0.96308172, 0.89715615, 0.83123058, 0.76530501]
            ),
        },
    },
    "2018": {
        0: {
            "nom": np.array(
                [0.86975122, 0.84051489, 0.81127857, 0.78204224, 0.75280592]
            ),
            "up": np.array([0.87484575, 0.84841907, 0.82199238, 0.79556569, 0.769139]),
            "down": np.array(
                [0.86465668, 0.83261072, 0.80056476, 0.7685188, 0.73647284]
            ),
        },
        1: {
            "nom": np.array(
                [0.89833224, 0.88032799, 0.86232374, 0.84431949, 0.82631524]
            ),
            "up": np.array([0.90637254, 0.89290176, 0.87943097, 0.86596019, 0.8524894]),
            "down": np.array(
                [0.89029193, 0.86775422, 0.8452165, 0.82267879, 0.80014108]
            ),
        },
        2: {
            "nom": np.array(
                [0.94639008, 0.92191403, 0.89743799, 0.87296194, 0.8484859]
            ),
            "up": np.array([0.958679, 0.94112899, 0.92357898, 0.90602896, 0.88847895]),
            "down": np.array(
                [0.93410115, 0.90269908, 0.871297, 0.83989492, 0.80849285]
            ),
        },
        3: {
            "nom": np.array([0.99796979, 0.96453794, 0.9311061, 0.89767425, 0.8642424]),
            "up": np.array(
                [1.03234984, 1.01797426, 1.00359869, 0.98922311, 0.97484753]
            ),
            "down": np.array(
                [0.96358975, 0.91110163, 0.8586135, 0.80612538, 0.75363726]
            ),
        },
        4: {
            "nom": np.array(
                [1.01068604, 0.97429003, 0.93789402, 0.90149802, 0.86510201]
            ),
            "up": np.array(
                [1.05358396, 1.04198126, 1.03037856, 1.01877586, 1.00717316]
            ),
            "down": np.array(
                [0.96778813, 0.90659881, 0.84540949, 0.78422017, 0.72303086]
            ),
        },
    },
}


def single_post_process(results, region, variable, samples, xss, nuisances, lumi):
    dout = {}
    year = lumi_to_year[int(lumi)]

    for histoName in samples:
        fix_nuisance_envelope = {}
        saved_nuisances = []
        is_data = samples[histoName].get("is_data", False) or samples[histoName].get(
            "postproc_is_data", False
        )
        for sample in samples[histoName]["samples"]:
            try:
                results[sample]["histos"][variable]
            except KeyError:
                print(f"Could not find key {sample} in {variable}")
            h = results[sample]["histos"][variable].copy()
            real_axis = list([slice(None) for _ in range(len(h.axes) - 2)])
            h = h[tuple(real_axis + [hist.loc(region), slice(None)])].copy()
            # renorm mcs
            if not is_data:
                h = renorm(h, xss[sample], results[sample]["sumw"], lumi)

            tmp_histo = h[tuple(real_axis + [hist.loc("nom")])].copy()
            hist_fold(tmp_histo, 3)
            if len(real_axis) > 1:
                tmp_histo = hist_unroll(tmp_histo)
            if "dphijj" in variable and "dnn" in variable:
                tmp_histo = make_new_histo(tmp_histo)
            key = f"{region}/{variable}/histo_{histoName}"
            if key not in dout:
                dout[key] = tmp_histo.copy()
            else:
                dout[key] += tmp_histo.copy()
            nom_histo = tmp_histo.copy()

            for nuis in nuisances:
                if nuisances[nuis].get("fake", False):
                    continue
                if nuisances[nuis]["type"] == "removeStat":
                    if eval(nuisances[nuis]["samples"].get(histoName, "0.0")) == 1.0:
                        key = f"{region}/{variable}/histo_{histoName}"
                        tmp_histo = dout[key]
                        a = tmp_histo.view(True)
                        a.variance = np.zeros_like(a.variance)
                        # print(key)
                        # print(tmp_histo.variances())
                    continue
                if nuisances[nuis]["type"] != "shape":
                    continue
                if histoName not in nuisances[nuis]["samples"]:
                    continue
                if nuisances[nuis]["kind"] in ["suffix", "weight"]:
                    nuis_name = nuisances[nuis]["name"]
                    saved_nuisances.append(nuis)
                    for tag in ["up", "down"]:
                        tmp_histo = h[
                            tuple(real_axis + [hist.loc(f"{nuis}_{tag}")])
                        ].copy()
                        hist_fold(tmp_histo, 3)
                        if len(real_axis) > 1:
                            tmp_histo = hist_unroll(tmp_histo)
                        if "dphijj" in variable and "dnn" in variable:
                            tmp_histo = make_new_histo(tmp_histo)

                        syst_key = f"{nuis_name}{tag.capitalize()}"
                        key = f"{region}/{variable}/histo_{histoName}_{syst_key}"

                        if nuisances[nuis].get("norm_map", None):
                            print(
                                "Unexpected norm map for non-theory variation",
                                file=sys.stderr,
                            )
                            # rescale = nuisances[nuis]["norm_map"][histoName][syst_key]
                            # a = tmp_histo.view(True)
                            # a.value = rescale * a.value
                            # a.variance = np.square(rescale) * a.variance

                        if key not in dout:
                            dout[key] = tmp_histo.copy()
                        else:
                            dout[key] += tmp_histo.copy()
                nuis_kind = nuisances[nuis]["kind"]
                if nuis_kind.endswith("envelope") or nuis_kind.endswith("square"):
                    variations = []
                    for nuis_histo in nuisances[nuis]["samples"][histoName]:
                        tmp_histo = h[tuple(real_axis + [hist.loc(nuis_histo)])].copy()
                        hist_fold(tmp_histo, 3)
                        if len(real_axis) > 1:
                            tmp_histo = hist_unroll(tmp_histo)

                        if "dphijj" in variable and "dnn" in variable:
                            tmp_histo = make_new_histo(tmp_histo)

                        norm_map = nuisances[nuis].get("norm_map", {})
                        if sample in norm_map and nuis_histo in norm_map[sample]:
                            rescale = norm_map[sample][nuis_histo]
                        if histoName in norm_map and nuis_histo in norm_map[histoName]:
                            rescale = norm_map[histoName][nuis_histo]
                        else:
                            rescale = 1.0

                        # rescale = (
                        #     nuisances[nuis]
                        #     .get("norm_map", {})
                        #     .get(sample, {})
                        #     .get(nuis_histo, 1.0)
                        # )

                        variations.append(tmp_histo.values() * rescale)
                    variations = np.array(variations)
                    if nuis not in fix_nuisance_envelope:
                        fix_nuisance_envelope[nuis] = variations.copy()
                    else:
                        fix_nuisance_envelope[nuis] += variations
        if not is_data:
            # fix empty bins
            nom_histo = dout[f"{region}/{variable}/histo_{histoName}"].copy()

            trim = False

            if "DY_hard" in histoName:
                bin = histoName.split("_")[-1]
                orig_nom_histo = nom_histo.values().copy()
                keys = [
                    key
                    for key in dout
                    if key.startswith(f"{region}/{variable}/histo_{histoName}_")
                ] + [f"{region}/{variable}/histo_{histoName}"]
                for key in keys:
                    tmp_histo = dout[key].copy()
                    a = tmp_histo.view()
                    if variable == "dphijj_reverse_flat_dnn" and year in corrections:
                        corr = corrections[year][int(bin)]["nom"].repeat(5)
                        # corr = np.ones_like(tmp_histo.values())
                    else:
                        corr = np.ones_like(tmp_histo.values())
                    a.value = corr * a.value
                    a.variance = np.square(corr) * a.variance
                    dout[key] = tmp_histo.copy()

                region_flav = region.split("_")[-1]
                for flav in ["ee", "mm"]:
                    if variable == "dphijj_reverse_flat_dnn" and region_flav == flav:
                        corr_up = corrections[year][int(bin)]["up"].repeat(5)
                        corr_do = corrections[year][int(bin)]["down"].repeat(5)
                    elif variable == "dphijj_reverse_flat_dnn" and region_flav != flav:
                        corr_up = corrections[year][int(bin)]["nom"].repeat(5)
                        corr_do = corrections[year][int(bin)]["nom"].repeat(5)
                    else:
                        corr_up = np.ones_like(nom_histo.values())
                        corr_do = np.ones_like(nom_histo.values())
                    # print(corr_up, corr_do)
                    histo = nom_histo.copy()
                    a = histo.view()
                    dy_key = histoName.split("_")[1]
                    a.value = corr_up * orig_nom_histo
                    a.variance = np.zeros_like(a.variance)
                    dout[
                        f"{region}/{variable}/histo_{histoName}_CMS_DY_{dy_key}_DNN_{bin}_{flav}_{year}_ShapeUp"
                    ] = histo.copy()

                    histo = nom_histo.copy()
                    a = histo.view()
                    a.value = corr_do * orig_nom_histo
                    a.variance = np.zeros_like(a.variance)
                    dout[
                        f"{region}/{variable}/histo_{histoName}_CMS_DY_{dy_key}_DNN_{bin}_{flav}_{year}_ShapeDown"
                    ] = histo.copy()

            # FIXME, add stat uncertainty to PU
            # if "DY" in histoName and "PU" in histoName:
            #     a = nom_histo.view()
            #     nom = a.value
            #     stat = np.sqrt(a.variance)
            #     a.variance = np.square(1.8 * stat)
            #     dout[f"{region}/{variable}/histo_{histoName}"] = nom_histo.copy()

            # if "DY" in histoName and "hard" in histoName:
            #     a = nom_histo.view()
            #     nom = a.value
            #     stat = np.sqrt(a.variance)
            #     a.variance = np.square(1.5 * stat)
            #     dout[f"{region}/{variable}/histo_{histoName}"] = nom_histo.copy()

            if trim:
                a = nom_histo.view()
                min_content = 1e-6
                bad_contents = a.value < min_content
                a.value = np.where(bad_contents, min_content, a.value)
                a.variance = np.where(bad_contents, min_content**2, a.variance)
                # if "DY" in histoName:
                #     # max stat unc can be 90%
                #     max_stat_unc = (0.9 * a.value.copy()) ** 2
                #     a.variance = np.where(
                #         a.variance > max_stat_unc, max_stat_unc, a.variance
                #     )
                dout[f"{region}/{variable}/histo_{histoName}"] = nom_histo.copy()

            for nuis in fix_nuisance_envelope:
                saved_nuisances.append(nuis)
                nuis_name = nuisances[nuis]["name"]
                variations = fix_nuisance_envelope[nuis]
                arrup = 0
                arrdo = 0
                nom_histo = dout[f"{region}/{variable}/histo_{histoName}"].copy()

                if nuisances[nuis]["kind"].endswith("envelope"):
                    # arrup = np.max(variations, axis=0)
                    # arrdo = np.min(variations, axis=0)
                    # if trim:
                    #     arrup = np.where(bad_contents, min_content, arrup)
                    #     arrdo = np.where(bad_contents, min_content, arrdo)
                    vnominal = nom_histo.values()
                    arrnom = np.tile(vnominal, (variations.shape[0], 1))
                    arrv = np.sqrt(np.mean(np.square(variations - arrnom), axis=0))
                    arrup = vnominal + arrv
                    arrdo = vnominal - arrv
                elif nuisances[nuis]["kind"].endswith("square"):
                    arrnom = np.tile(nom_histo.values(), (variations.shape[0], 1))
                    arrv = np.sqrt(np.sum(np.square(variations - arrnom), axis=0))
                    if trim:
                        arrv = np.where(bad_contents, min_content, arrv)
                    arrup = nom_histo.values() + arrv
                    arrdo = nom_histo.values() - arrv
                hists = {}
                hists["Up"] = nom_histo.copy()
                a = hists["Up"].view()
                a.value = arrup

                hists["Down"] = nom_histo.copy()
                a = hists["Down"].view()
                a.value = arrdo

                for tag in ["Up", "Down"]:
                    key = f"{region}/{variable}/histo_{histoName}_{nuis_name}{tag}"
                    tmp_histo = hists[tag]
                    dout[key] = tmp_histo.copy()

            if trim:
                for nuis in saved_nuisances:
                    nuis_name = nuisances[nuis]["name"]
                    for tag in ["Up", "Down"]:
                        key = f"{region}/{variable}/histo_{histoName}_{nuis_name}{tag}"
                        histo = dout[key].copy()
                        a = histo.view()
                        a.value = np.where(bad_contents, min_content, a.value)
                        # trim DY to stat
                        if "DY" in histoName:
                            nom_vals = nom_histo.values()
                            nom_err = np.sqrt(nom_histo.variances())
                            a.value = np.where(
                                a.value > nom_vals + nom_err,
                                nom_vals + nom_err,
                                a.value,
                            )
                            a.value = np.where(
                                a.value < nom_vals - nom_err,
                                nom_vals - nom_err,
                                a.value,
                            )
                        dout[key] = histo.copy()

            # reduce_stat = "DY" in histoName
            # if reduce_stat:
            #     a = nom_histo.view()
            #     a.variance = a.variance / 2**2
            #     dout[f"{region}/{variable}/histo_{histoName}"] = nom_histo.copy()
            #     # for nuis in saved_nuisances:
            #     #     for tag in ["Up", "Down"]:
            #     #         key = f"{region}/{variable}/histo_{histoName}_{nuisances[nuis]['name']}{tag}"
            #     #         h = dout[key].copy()
            #     #         a = h.view()
            #     #         delta = a.value - nom_histo.values()
            #     #         a.value = nom_histo.values() + delta / 2
            #     #         dout[key] = h.copy()

            # symmetrize jer
            for nuis in ["JER"] + [f"JER_{i}" for i in range(2, 6)]:
                if nuis not in nuisances:
                    continue
                if histoName not in nuisances[nuis]["samples"]:
                    continue
                nuis_name = nuisances[nuis]["name"]
                nom_histo = dout[f"{region}/{variable}/histo_{histoName}"].copy()
                nom = nom_histo.values()
                key = f"{region}/{variable}/histo_{histoName}_{nuis_name}Up"
                tmp_histo = dout[key].copy()
                up = tmp_histo.values()
                key = f"{region}/{variable}/histo_{histoName}_{nuis_name}Down"
                tmp_histo = dout[key].copy()
                down = tmp_histo.values()

                delta = np.maximum(abs(up - nom), abs(nom - down))
                # FIXME limit between 0 and 2*nom the down and up ?
                delta = np.where(delta < 0.0, 0.0, np.where(delta > nom, nom, delta))

                # if "DY_PU" in histoName:
                #     stat_err = np.sqrt(nom_histo.variances())
                #     delta = delta / 2.0
                #     delta = np.where(delta > stat_err, stat_err, delta)

                hs = {}
                hs["Up"] = nom + delta
                hs["Down"] = nom - delta

                for tag in ["Up", "Down"]:
                    key = f"{region}/{variable}/histo_{histoName}_{nuis_name}{tag}"
                    tmp_histo = dout[key].copy()
                    a = tmp_histo.view()
                    a.value = hs[tag].copy()
                    dout[key] = tmp_histo.copy()

            # # FIXME DY PU 0 low stat!
            # # if histoName == "DY_PU_4":
            # if "DY_PU" in histoName:
            #     a = nom_histo.view()
            #     a.value = np.where(a.value <= 0.0, 1e-3, a.value)
            #     a.variance = np.square(a.value * 0.8)
            #     a.variance = np.where(
            #         np.sqrt(a.variance) / a.value > 0.8,
            #         np.square(a.value * 0.8),
            #         a.variance,
            #     )
            #     dout[f"{region}/{variable}/histo_{histoName}"] = nom_histo.copy()
            #     for key in dout:
            #         if key.startswith(f"{region}/{variable}/histo_{histoName}_"):
            #             # print("fixing", key)
            #             tmp_histo = dout[key].copy()
            #             a = tmp_histo.view()
            #             max_delta = 0.8
            #             a.value = np.where(
            #                 a.value > (1 + max_delta) * nom_histo.values(),
            #                 (1 + max_delta) * nom_histo.values(),
            #                 a.value,
            #             )
            #             a.value = np.where(
            #                 a.value < (1 - max_delta) * nom_histo.values(),
            #                 (1 - max_delta) * nom_histo.values(),
            #                 a.value,
            #             )
            #             dout[key] = tmp_histo.copy()

    for key in dout:
        integral = np.sum(dout[key].values())
        if integral <= 0:
            if "top_cr" not in key:
                print("Negative integral", key, integral)
            tmp_histo = dout[key].copy()
            a = tmp_histo.view()
            val = np.zeros_like(a.value)
            val[0] = 1e-3
            a.value = val.copy()
            a.variance = val.copy()
            dout[key] = tmp_histo.copy()

    return dout


def make_events(dout, region, variable):
    keys = list(dout.keys())
    keys = list(filter(lambda x: f"{region}/{variable}/histo_" in x, keys))
    for key in keys:
        h = hist.Hist(
            hist.axis.Regular(1, 0.0, 1.0, name="Events"), hist.storage.Weight()
        )
        a = h.view()
        h_to_merge = dout[key].copy()
        a.value = np.sum(h_to_merge.values())
        a.variance = np.sum(h_to_merge.variances())
        hname = key.split("/")[-1]
        dout[f"{region}/events/{hname}"] = h.copy()
    return dout


def post_process(results, regions, variables, samples, xss, nuisances, lumi):
    print("Start converting histograms")

    cpus = 10

    good_regions = regions
    # good_regions = ["sr_0p8_1p0_nogap", "sr_0p8_1p0_gap", "dypu_0p8_1p0", "top_0p8_1p0"]
    # good_regions = ["sr_0p9_1p0_nogap", "sr_0p9_1p0_gap", "dypu_0p9_1p0", "top_0p9_1p0"]
    # good_regions = [
    #     "sr_inc_ee",
    #     "sr_inc_mm",
    #     "dypu_cr_ee",
    #     "dypu_cr_mm",
    #     "top_cr_ee",
    #     "top_cr_mm",
    # ]
    # good_regions = list(filter(lambda x: "high" in x, good_regions))

    # good_regions = [
    #     f"{reg}_{cat}"
    #     for reg in ["sr_inc", "dypu_cr", "top_cr"]
    #     for cat in ["ee", "mm"]
    # ]

    good_variables = []
    for variable in variables:
        # if variable == "events":
        #     continue
        if "axis" not in variables[variable]:
            continue

        # if variable not in [
        #     # "dphijj_reverse_flat_dnn",
        #     # "dphijj_reverse_dnn_many",
        #     # "dphijj_reverse",
        #     # "dphijj_reverse_morebins",
        #     # "dnn_fits",
        #     "dnn",
        #     # "MET_fits",
        #     # "mjj",
        #     # "ptll_unfold",
        #     "mjj_unfold",
        #     # "dphijj_unfold",
        #     # "dphill_unfold",
        #     # "detajj_unfold",
        #     # "ptjj_unfold",
        #     # "ptj3_unfold",
        #     # "add_HT_unfold",
        #     # "ptj1",
        #     # "ptj2",
        #     # "ptll",
        #     # "mjj",
        #     # "detajj_fits",
        #     # # "ptll",
        #     # # "dr_jj",
        #     # # "ptj1",
        #     # # "ptj2",
        #     # # "dnn_fits",
        #     # "dphijj",
        #     # "Zeppenfeld_Z",
        #     # "pt_balance",
        #     # "nvertices",
        #     # "zepp_r",
        #     # "dnn",
        #     # "mjj_atlas",
        # ]:
        #     continue

        good_variables.append(variable)

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus) as executor:
        tasks = []
        print("start post-proc in parallel")
        for region in good_regions:
            for variable in good_variables:
                print("submitting", region, variable)
                tasks.append(
                    executor.submit(
                        single_post_process,
                        results,
                        region,
                        variable,
                        samples,
                        xss,
                        nuisances,
                        lumi,
                    )
                )
        concurrent.futures.wait(tasks)
        print("done post-proc in parallel")
        results = []
        for task in tasks:
            results.append(task.result())
        dout = add_dict_iterable(results)

    # for region in regions:
    #     dout = make_events(dout, region, good_variables[0])

    # dout = make_events(dout, region, "dphijj_reverse_flat_dnn")

    # # merge regions from ee an mm
    # for region in ["sr_inc", "dypu_cr", "top_cr"]:
    #     tot_keys = []
    #     for cat in ["ee", "mm"]:
    #         keys = list(filter(lambda k: k.startswith(f"{region}_{cat}/"), dout.keys()))
    #         keys = list(map(lambda k: k.replace(f"{region}_{cat}/", ""), keys))
    #         tot_keys += keys
    #     tot_keys = list(set(tot_keys))

    #     for key in tot_keys:
    #         # histo = key.split("/")[1:]
    #         new_key = f"{region}/{key}"

    #         for cat in ["ee", "mm"]:
    #             old_key = f"{region}_{cat}/{key}"
    #             if old_key not in dout:
    #                 # old_key = f"{region}_{cat}/{key}/histo_{key}"
    #                 print(f"missing {old_key}")
    #             if new_key not in dout:
    #                 dout[new_key] = dout[old_key].copy()
    #             else:
    #                 dout[new_key] += dout[old_key].copy()

    #             del dout[old_key]

    good_variables += ["events"]

    # print(dout.keys())

    # normalize signal
    # for region in good_regions:
    #     for variable in good_variables:
    #         dout[f"{region}/{variable}/histo_Zjj_fiducial_i"]

    # # FIXME norm QCD scale for DY Jets

    # nuisances_fix = ["QCDScale_DY", "PS_FSR_DY", "PS_ISR_DY", "PDF_DY"]
    # nuisances_fix = ["QCDScale_DY", "PS_ISR_DY", "PDF_DY"]
    # samples_fix = [f"DY_{key}_{bin}" for key in ["hard", "PU"] for bin in range(5)]

    # for histoName in samples_fix:
    #     nom_sum = 0
    #     nuis_sum = {nuis: {"Up": 0, "Down": 0} for nuis in nuisances_fix}
    #     for region in regions:
    #         variable = "events"
    #         key = f"{region}/{variable}/histo_{histoName}"
    #         nom_sum += np.sum(dout[key].values())
    #         for nuis in nuisances_fix:
    #             for tag in ["Up", "Down"]:
    #                 key = f"{region}/{variable}/histo_{histoName}_{nuis}{tag}"
    #                 nuis_sum[nuis][tag] += np.sum(dout[key].values())

    #     print(histoName)
    #     print(f"Nominal sum: {nom_sum}")
    #     print(f"Up sum: {nuis_sum['QCDScale_DY']['Up']}")
    #     print(f"Down sum: {nuis_sum['QCDScale_DY']['Down']}")

    #     for region in good_regions:
    #         for variable in good_variables:
    #             for nuis in nuisances_fix:
    #                 for tag in ["Up", "Down"]:
    #                     key = f"{region}/{variable}/histo_{histoName}_{nuis}{tag}"
    #                     tmp_histo = dout[key]
    #                     a = tmp_histo.view()
    #                     a.value = a.value * (nom_sum / nuis_sum[nuis][tag])
    #                     dout[key] = tmp_histo.copy()

    # nom_sum = 0
    # nuis_sum = {nuis: {"Up": 0, "Down": 0} for nuis in nuisances_fix}
    # for region in regions:
    #     variable = "dphijj_reverse_flat_dnn"
    #     for histoName in samples_fix:
    #         key = f"{region}/{variable}/histo_{histoName}"
    #         nom_sum += np.sum(dout[key].values())
    #         for nuis in nuisances_fix:
    #             for tag in ["Up", "Down"]:
    #                 key = f"{region}/{variable}/histo_{histoName}_{nuis}{tag}"
    #                 nuis_sum[nuis][tag] += np.sum(dout[key].values())

    # print(f"Nominal sum: {nom_sum}")
    # print(f"Up sum: {nuis_sum['QCDScale_DY']['Up']}")
    # print(f"Down sum: {nuis_sum['QCDScale_DY']['Down']}")

    print("start saving in root file")
    fout = uproot.recreate("histos.root")
    for key in dout:
        fout[key] = dout[key]
    fout.close()


def main():
    analysis_dict = get_analysis_dict()
    year = analysis_dict["year"]
    lumi = analysis_dict["lumi"]
    datasets = analysis_dict["datasets"]
    samples = analysis_dict["samples"]
    nuisances = analysis_dict["nuisances"]
    regions = analysis_dict["regions"]
    variables = analysis_dict["variables"]

    with open(f"{path_fw}/data/{year}/samples/samples.json") as file:
        samples_xs = json.load(file)

    # samples["DY-Inc"]["is_data"] = False

    # xs = [
    #     0.26427865009355,
    #     1.0418643158816585,
    #     1.1083723860511394,
    #     1.1546134964933237,
    #     1.1831173691113572,
    #     1.2001124763697817,
    # ]
    # for bin in range(0, 6):
    #     samples_xs["samples"][f"DYJetsToLL_Pt_{bin}"]["xsec"] += f"* {xs[bin]}"

    # xs = [5896, 389.4, 92.35, 3.506, 0.4758, 0.04492]
    # for bin in range(0, 6):
    #     samples_xs["samples"][f"DYJetsToLL_Pt_{bin}"]["xsec"] = f"{xs[bin]}"
    # samples_xs["samples"][f"DYJetsToLL_Pt_0"]["xsec"] += "* 0.2526"

    # samples_xs["samples"][f"DYJetsToLL_Pt_0"]["xsec"] += "*0.26"
    # samples_xs["samples"][f"DYJetsToLL_Pt_0"]["xsec"] += "*0.26"
    # for bin in range(1, 6):
    #     # samples_xs["samples"][f"DYJetsToLL_Pt_{bin}"]["xsec"] = f"{xs[bin]}"
    #     samples_xs["samples"][f"DYJetsToLL_Pt_{bin}"]["xsec"] += "*1.2"
    # samples_xs["samples"]["DYJetsToLL_M-50"]["xsec"] = "6399.0"

    xss = {}
    for dataset in datasets:
        if datasets[dataset].get("is_data", False) or datasets[dataset].get(
            "postproc_is_data", False
        ):
            continue
        key = datasets[dataset]["files"]
        print(key)

        # # # FIXME
        # if "DY" in dataset:
        #     samples_xs["samples"][key]["xsec"] += "*0.5"

        if "subsamples" in datasets[dataset]:
            for sub in datasets[dataset]["subsamples"]:
                flat_dataset = f"{dataset}_{sub}"
                # if fnmatch.fnmatch(flat_dataset, "DY*PU"):
                #     xss[flat_dataset] = (
                #         eval(samples_xs["samples"][key]["xsec"]) * 7.7647e-01
                #     )
                # elif fnmatch.fnmatch(flat_dataset, "DY*hard"):
                #     xss[flat_dataset] = (
                #         eval(samples_xs["samples"][key]["xsec"]) * 9.2941e-01
                #     )
                # else:
                #     xss[flat_dataset] = eval(samples_xs["samples"][key]["xsec"])
                xss[flat_dataset] = eval(samples_xs["samples"][key]["xsec"])
                print(flat_dataset, xss[flat_dataset])
        else:
            flat_dataset = dataset
            xss[flat_dataset] = eval(samples_xs["samples"][key]["xsec"])

    print(xss)
    results = read_chunks("condor/results_merged_new.pkl")
    print(results.keys())
    # sys.exit()
    post_process(results, regions, variables, samples, xss, nuisances, lumi)


if __name__ == "__main__":
    main()