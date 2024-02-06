
begin
;-------Options--------


  undef("read_ocean_land_colormap")
  function read_ocean_land_colormap(num_ocean)
  local cmap_ocn, cmap_lnd
  begin
    cmap_ocn = read_colormap_file("GMT_ocean")
    cmap_lnd = read_colormap_file("OceanLakeLandSnow")
    newcmap = array_append_record(cmap_ocn(0:num_ocean-1,:),cmap_lnd(2::2,:),0)
    return(newcmap)
  end
  
  undef("calc_levels_and_colors")
  function calc_levels_and_colors(wks,emin,emax,split_elev,num_ocean_values)
  local start_ocean, ocean_range, land_range, olevels, llevels, nol, nll, clen
  begin
    cmap = read_ocean_land_colormap(num_ocean_values)
    clen = dimsizes(cmap(:,0))
  
    start_ocean = 0
    end_ocean   = num_ocean_values-1
    start_land  = end_ocean+1
    ocean_range = end_ocean-start_ocean+1
    land_range  = clen-start_land+1
    olevels     = fspan(emin,split_elev,ocean_range)
    llevels     = fspan(split_elev,emax,land_range)
    nol         = dimsizes(olevels)
    nll         = dimsizes(llevels)
    levels      = new((nol-1)+(nll-2),float)
    levels(0:nol-2) = olevels(1:)
    levels(nol-1:)  = llevels(1:nll-2)
    return([/levels,cmap/])
  end
  

    wks = gsn_open_wks("pdf","Figure1-a111") ; send graphics to PNG file
  
    topo_filename = "data/RR_ANA_FOR-2021071200_202107112350_202107112348.nc"

    f      = addfile(topo_filename, "r")
    elev    = f->topo
    lat1    = f->lat
    lon1    = f->lon
    elev@lat2d = lat1
    elev@lon2d = lon1

    infile = "location.csv"
    lines  = asciiread (infile, -1, "string")  ; -1 means read whole file
    strs   = str_split_csv(lines, ",", 0)    
    nums   = tofloat (strs)                    ; _FillValue = 9.96921e+36

    arr = (/30.,40.,50./)   
    colors = (/"lightskyblue1","springgreen1","lightgoldenrod","lightsalmon"/)
    labels = new(dimsizes(arr)+1,string)
    lat  = nums(:,0)         
    lon  = nums(:,1)
    R    = nums(:,2)
  
  
    num_distinct_markers = dimsizes(arr)+1     
    lat_new = new((/num_distinct_markers,dimsizes(R)/),float,-999)
    lon_new = new((/num_distinct_markers,dimsizes(R)/),float,-999)
  
    do i = 0, num_distinct_markers-1
      if (i.eq.0) then
        indexes = ind(R.lt.arr(0))
        labels(i) = "x < " + arr(0)
      end if
      if (i.eq.num_distinct_markers-1) then
        indexes = ind(R.ge.max(arr))
        labels(i) = "x >= " + max(arr)
      end if
      if (i.gt.0.and.i.lt.num_distinct_markers-1) then       
        indexes = ind(R.ge.arr(i-1).and.R.lt.arr(i))
        labels(i) = arr(i-1) + " <= x < " + arr(i)
      end if
  
      if (.not.any(ismissing(indexes))) then 
        npts_range = dimsizes(indexes)   ; # of points in this range.
        lat_new(i,0:npts_range-1) = lat(indexes)
        lon_new(i,0:npts_range-1) = lon(indexes)
      end if
      delete(indexes)            ; Necessary b/c "indexes" may be a different
                                 ; size next time.
    end do
  
    res                    = True

    res@txFont           = 25
    res@vpWidthF         = 0.6            ; Change the aspect ratio, but 
    res@vpHeightF        = 0.6           ; make plot as large as possible.


    ;res@gsnMaximize    = True
    res@gsnRightString = ""
    res@gsnLeftString = ""
    res@tmXBMajorOutwardLengthF = -0.001
    res@tmXBMinorOutwardLengthF = -0.001
    res@tmYLMajorOutwardLengthF = -0.001
    res@tmYLMinorOutwardLengthF = -0.001
    res@tmYRMajorOutwardLengthF = -0.001
    res@tmYRMinorOutwardLengthF = -0.001
  ;  res@tiYAxisFontHeightF = 0.014


    res@gsnFrame           = False        ; will draw in panel later
    res@gsnDraw            = False
    res@cnFillOn           = True     ; turn on contour fill
    res@cnFillMode         = "MeshFill"
    res@cnLinesOn          = False    ; turn off contour lines
    res@cnLineLabelsOn     = False    ; turn off line labels
    res@cnInfoLabelOn      = False    ; turn off info label
    res@lbBoxLinesOn       = False    ; turn off labelbar box lines
    res@lbOrientation      = "Vertical"
    res@lbTitleString      = "~F25~Elevation (m)"
    res@lbLabelFontThicknessF = 0.01
    res@lbLabelFont        = 25
    res@lbTitleFontHeightF = 0.016
  ;---Calculate "nice" contour levels, and create a color map to match
    split_elev               = 200; -68    ; meters
    num_ocean_colors         = 43
    levels_and_colors        = calc_levels_and_colors(wks,min(elev),max(elev),split_elev,num_ocean_colors)
    res@cnLevelSelectionMode = "ExplicitLevels"
    res@cnLevels             = levels_and_colors[0]
    res@cnFillPalette        = levels_and_colors[1]
    res@gsnAddCyclic         = False           ; don't add longitude cyclic point
    res@mpFillOn                    = False
    res@pmTickMarkDisplayMode       = "Always"   ; Nicer map labels
    res@mpGeophysicalLineThicknessF = 2          ; Thicker map outlines
    res@mpOutlineOn       = False

    res@mpMinLatF         = min(elev@lat2d)+0.1
    res@mpMaxLatF         = max(elev@lat2d)-0.1
    res@mpMinLonF         = min(elev@lon2d)+0.5
    res@mpMaxLonF         = max(elev@lon2d)-0.5
  ;---Generate our own labels for the labelbar
    nlevels           = dimsizes(res@cnLevels)
    labels1            = new(nlevels+2,string)
    labels1            = ""                 ; Blank out all but
    ii                = ind(res@cnLevels.eq.split_elev)+1
    labels1(0)         = "" + min(elev)     ; First,
    labels1(nlevels+1) = "3000"    ; last, and 
    labels1(ii)         = "" + split_elev   ; middle labels
  
    res@lbLabelAutoStride = False            ; This will make sure every labelbar 
    res@lbLabelAlignment = "ExternalEdges"   ; can potentially be labeled.
    res@lbLabelStrings   = labels1

    plot = gsn_csm_contour_map(wks,elev,res)


    lnres                   = True
    lnres@gsLineThicknessF  = 5.0
    lnres@gsLineColor       = "grey"
    ind_adm1 = gsn_add_shapefile_polylines(wks,plot,"../../map/jjj_boundary.shp",lnres)
    draw(plot)

    gsres               = True
    gsres@gsMarkerIndex = 16          ; Use filled dots for markers.
    gsres@gsMarkerSizeF = 0.004
    txres               = True
    txres@txFontHeightF = 0.018
    txres@txFontColor   = "black"
    txres@txFont        = 25

  xleg = (/0.57,0.57,0.82,0.82/)-0.3   ; Location of
  xtxt = (/0.66,0.66,0.91,0.91/)-0.3   ; legend markers
  yleg = (/0.22,0.17,0.22,0.17/)   ; and text
  ytxt = (/0.22,0.17,0.22,0.17/)   ; strings.

  do i = 0, num_distinct_markers-1
    if (.not.ismissing(lat_new(i,0)))
      gsres@gsMarkerColor      = colors(i)
      gsres@gsMarkerThicknessF = 0.7*(i+1)
      gsn_polymarker(wks,plot,lon_new(i,:),lat_new(i,:),gsres)
      gsn_polymarker_ndc(wks,          xleg(i),yleg(i),gsres)
      gsn_text_ndc      (wks,labels(i),xtxt(i),ytxt(i),txres)
    end if
  end do
  frame(wks)    


end     
