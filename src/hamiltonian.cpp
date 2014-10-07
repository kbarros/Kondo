

/*
trait KondoHamiltonian {
    val numLatticeSites: Int
    val hoppingMatrix: PackedSparse[S]
    val e_max: R
    val e_min: R
    val J_H: R // Hund coupling
    
    def setField(s: String)
    
    def latticePositions: Array[Vec3]
    
    val e_avg   = (e_max + e_min)/2
    val e_scale = (e_max - e_min)/2
    
    // convert from physical energy units to scaled energy units
    def scaleEnergy(x: R): R = {
        (x - e_avg) / e_scale
    }
    
    def pauli = Array[S#A] (
                            0, 1,
                            1, 0,
                            
                            0, I, // visually transposed, due to row major ordering
                            -I, 0,
                            
                            1, 0,
                            0, -1
                            )
    
    // these fields are lazy because we must wait for subclass
    // to provide numLatticeSites
    lazy val field: Array[R] = {
        val ret = new Array[R](3*numLatticeSites)
        setFieldFerro(ret)
        ret
    }
    lazy val delField: Array[R] = Array.fill(3*numLatticeSites)(0)
    lazy val matrix = {
        val ret = sparse(2*numLatticeSites, 2*numLatticeSites): HashSparse[S]
        fillMatrix(ret)
        ret.toPacked
    }
    lazy val delMatrix: PackedSparse[S] = {
        matrix.duplicate.clear
    }
    
    def pauliIndex(sp1: Int, sp2: Int, d: Int): Int = {
        sp1 + sp2*(2) + d*(2*2)
    }
    
    def fieldIndex(d: Int, li: Int): Int = {
        d + li*(3)
    }
    
    def matrixIndex(sp: Int, li: Int): Int = {
        sp + li*(2)
    }
    
    def getSpin(field: Array[R], li: Int) = {
        Vec3(field(fieldIndex(0, li)),
             field(fieldIndex(1, li)),
             field(fieldIndex(2, li)))
    }
    
    def setSpin(field: Array[R], li: Int, s: Vec3) {
        field(fieldIndex(0, li)) = s.x
        field(fieldIndex(1, li)) = s.y
        field(fieldIndex(2, li)) = s.z
    }
    
    def setFieldFerro(field: Array[R]) {
        field.transform(_ => 1.0)
        normalizeField(field)
    }
    
    def setFieldRandom(field: Array[R], rand: util.Random) {
        field.transform(_ => rand.nextGaussian())
        normalizeField(field)
    }
    
    def normalizeField(field: Array[R], validate: Boolean = false) {
        for (li <- 0 until numLatticeSites) {
            var acc = 0d
            for (d <- 0 until 3) {
                acc += field(fieldIndex(d, li)).abs2
            }
            acc = math.sqrt(acc)
            if (validate && !(acc > 0.95 && acc < 1.05))
                println("Vector magnitude %g deviates too far from normalization".format(acc))
                for (d <- 0 until 3) {
                    field(fieldIndex(d, li)) /= acc
                }
        }
    }
    
    // remove component of dS that is parallel to field S
    def projectTangentField(S: Array[R], dS: Array[R]) {
        for (li <- 0 until numLatticeSites) {
            var s_dot_s = 0d
            var s_dot_ds = 0d
            for (d <- 0 until 3) {
                val i = fieldIndex(d, li)
                s_dot_s  += S(i)*S(i)
                s_dot_ds += S(i)*dS(i)
            }
            val alpha = s_dot_ds / s_dot_s
            for (d <- 0 until 3) {
                val i = fieldIndex(d, li)
                dS(i) -= alpha * S(i)
            }
        }
    }
    
    def fillMatrix[M[s <: Scalar] <: Sparse[s, M]](m: M[S]) {
        m.clear()
        
        // hopping term
        for ((i, j) <- hoppingMatrix.definedIndices) {
            m(i, j) += hoppingMatrix(i, j)
        }
        
        // hund coupling term
        for (li <- 0 until numLatticeSites;
             sp1 <- 0 until 2;
             sp2 <- 0 until 2) {
            var coupling = 0: S#A
            for (d <- 0 until 3) {
                coupling += pauli(pauliIndex(sp1, sp2, d)) * field(fieldIndex(d, li))
            }
            val i = matrixIndex(sp1, li)
            val j = matrixIndex(sp2, li)
            m(i, j) = -J_H * coupling
        }
        
        // scale matrix appropriately so that eigenvalues lie between -1 and +1
        for (i <- 0 until m.numRows) { m(i,i) -= e_avg }
        m /= e_scale
        
        //    // Make sure hamiltonian is hermitian
        //    val H = m.toDense
        //    require((H - H.dag).norm2.abs < 1e-6, "Found non-hermitian hamiltonian!")
    }
    
    // Use chain rule to transform derivative wrt matrix elements dF/dH, into derivative wrt spin indices
    //   dF/dS = dF/dH dH/dS
    // In both factors, H is assumed to be dimensionless (scaled energy). If F is also dimensionless, then it
    // may be desired to multiply the final result by the energy scale.
    def fieldDerivative(dFdH: PackedSparse[S], dFdS: Array[R]) {
        // loop over all lattice sites and vector indices
        for (li <- 0 until numLatticeSites;
             d <- 0 until 3) {
            
            var dCoupling: S#A = 0
            for (sp1 <- 0 until 2;
                 sp2 <- 0 until 2) {
                val i = matrixIndex(sp1, li)
                val j = matrixIndex(sp2, li)
                dCoupling += dFdH(i, j) * pauli(pauliIndex(sp1, sp2, d))
            }
            require(math.abs(dCoupling.im) < 1e-5, "Imaginary part of field derivative non-zero: " + dCoupling.im)
            dFdS(fieldIndex(d, li)) = -J_H * dCoupling.re
        }
        
        // the derivative is perpendicular to the direction of S, due to constraint |S|=1 
        projectTangentField(field, dFdS)
        
        // properly scale the factor dH/dS, corresponding to scaled H
        dFdS.transform(_ / e_scale)
    }
    }
    
*/