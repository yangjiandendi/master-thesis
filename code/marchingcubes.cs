public static void IsoFaces(ref GridCell cell, float surfacelevel)
    {
        // Parameters:   
            //  cell = GridCell to analyze for triangles
            //  surfacelevel = GridPoint value to be less than to be considered on points

        // Returns:
            //  gridcell will have the number of triangles and triangles array defined

        cell.ClearCalculations();

        // step 1. determine cell config, corners below the surface of the mesh (inside / outside mesh) 
        cell.config = 0;
        if (cell.p[0].Value < surfacelevel)
            Bits.SetBit(ref cell.config, 0);    // A =0
        if (cell.p[1].Value < surfacelevel)
            Bits.SetBit(ref cell.config, 1);    // B =1
        if (cell.p[2].Value < surfacelevel)
            Bits.SetBit(ref cell.config, 2);    // C =2
        if (cell.p[3].Value < surfacelevel)
            Bits.SetBit(ref cell.config, 3);    // D =3
        if (cell.p[4].Value < surfacelevel)
            Bits.SetBit(ref cell.config, 4);    // E =4
        if (cell.p[5].Value < surfacelevel)
            Bits.SetBit(ref cell.config, 5);    // F =5
        if (cell.p[6].Value < surfacelevel)
            Bits.SetBit(ref cell.config, 6);    // G =6
        if (cell.p[7].Value < surfacelevel)
            Bits.SetBit(ref cell.config, 7);    // H =7

        if (edgeTable[cell.config] == 0)    //cell is entirely inside/outside mesh surface (make no triangles)
            return;

        // step 2. determine interpolated edge point positions (where applicable)
        if (Bits.isSet(edgeTable[cell.config], 0) == true) 
            cell.edgepoint[0] = InterpolateEdgePosition(surfacelevel, cell.p[0], cell.p[1]);
        if (Bits.isSet(edgeTable[cell.config], 1) == true) 
            cell.edgepoint[1] = InterpolateEdgePosition(surfacelevel, cell.p[1], cell.p[2]);
        if (Bits.isSet(edgeTable[cell.config], 2) == true) 
            cell.edgepoint[2] = InterpolateEdgePosition(surfacelevel, cell.p[2], cell.p[3]);
        if (Bits.isSet(edgeTable[cell.config], 3) == true) 
            cell.edgepoint[3] = InterpolateEdgePosition(surfacelevel, cell.p[3], cell.p[0]);
        if (Bits.isSet(edgeTable[cell.config], 4) == true) 
            cell.edgepoint[4] = InterpolateEdgePosition(surfacelevel, cell.p[4], cell.p[5]);
        if (Bits.isSet(edgeTable[cell.config], 5) == true) 
            cell.edgepoint[5] = InterpolateEdgePosition(surfacelevel, cell.p[5], cell.p[6]);
        if (Bits.isSet(edgeTable[cell.config], 6) == true) 
            cell.edgepoint[6] = InterpolateEdgePosition(surfacelevel, cell.p[6], cell.p[7]);
        if (Bits.isSet(edgeTable[cell.config], 7) == true) 
            cell.edgepoint[7] = InterpolateEdgePosition(surfacelevel, cell.p[7], cell.p[4]);
        if (Bits.isSet(edgeTable[cell.config], 8) == true) 
            cell.edgepoint[8] = InterpolateEdgePosition(surfacelevel, cell.p[0], cell.p[4]);
        if (Bits.isSet(edgeTable[cell.config], 9) == true) 
            cell.edgepoint[9] = InterpolateEdgePosition(surfacelevel, cell.p[1], cell.p[5]);
        if (Bits.isSet(edgeTable[cell.config], 10) == true) 
            cell.edgepoint[10] = InterpolateEdgePosition(surfacelevel, cell.p[2], cell.p[6]);
        if (Bits.isSet(edgeTable[cell.config], 11) == true) 
            cell.edgepoint[11] = InterpolateEdgePosition(surfacelevel, cell.p[3], cell.p[7]);

        // step 3. determine triangles (iso faces)
        for (int i = 0; triangleTable[cell.config,i] != -1; i += 3)
        {
            cell.triangle[cell.numtriangles].p[0] = cell.edgepoint[triangleTable[cell.config, i]];
            cell.triangle[cell.numtriangles].p[1] = cell.edgepoint[triangleTable[cell.config, i + 1]];
            cell.triangle[cell.numtriangles].p[2] = cell.edgepoint[triangleTable[cell.config, i + 2]];
            cell.numtriangles++;
        }
    }