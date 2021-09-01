public void MakeProceduralGrid3()
    {
        //create space for vertices and triangles
        vertices = new Vector3[Num_in_V_Direction * Num_in_U_Direction];
        triangles = new int[(Num_in_V_Direction - 1) * 2 * 3 * (Num_in_U_Direction - 1)];

        // give Surface point positions to vertices
        for (int i = 0; i < Surfacepointpositions.Count; i++)
        {
            vertices[i] = Surfacepointpositions[i];
        }

        // calculate triangles in U and V direction
        for (int j = 0; j < Num_in_U_Direction - 1; j++)
        {
            for (int i = 0; i < Num_in_V_Direction - 1; i++)
            {
                triangles[i * 3 + j * (Num_in_V_Direction - 1) * 2 * 3] = i + j * Num_in_V_Direction;
                triangles[i * 3 + 1 + j * (Num_in_V_Direction - 1) * 2 * 3] = i + 1 + j * Num_in_V_Direction;
                triangles[i * 3 + 2 + j * (Num_in_V_Direction - 1) * 2 * 3] = i + Num_in_V_Direction + j * Num_in_V_Direction;
            }

            for (int i = 0; i < (Num_in_V_Direction - 1); i++)
            {
                triangles[((Num_in_V_Direction - 2) * 3 + 3) + i * 3 + j * (Num_in_V_Direction - 1) * 2 * 3] = i + Num_in_V_Direction + 1 + j * Num_in_V_Direction;
                triangles[((Num_in_V_Direction - 2) * 3 + 3) + i * 3 + 1 + j * (Num_in_V_Direction - 1) * 2 * 3] = i + Num_in_V_Direction + j * Num_in_V_Direction;
                triangles[((Num_in_V_Direction - 2) * 3 + 3) + i * 3 + 2 + j * (Num_in_V_Direction - 1) * 2 * 3] = i + 1 + j * Num_in_V_Direction;
            }
        }
    }