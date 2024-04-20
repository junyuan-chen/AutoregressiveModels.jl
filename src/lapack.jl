# Define customized in-placed methods of SVD that are non-allocating
# Use liblapack instead of libblastrampoline for compatibility with Julia v1.6

struct SDDcache{TF<:Union{Float64, Float32}}
    U::Matrix{TF}
    VT::Matrix{TF}
    S::Vector{TF}
    work::Vector{TF}
    iwork::Vector{BlasInt}
    info::Base.RefValue{BlasInt}
end

function SDDcache(A::AbstractMatrix{TF}) where TF<:Union{Float64, Float32}
    m, n = size(A)
    minmn = min(m, n)
    U = Matrix{TF}(undef, m, minmn)
    VT = Matrix{TF}(undef, minmn, n)
    S = Vector{TF}(undef, minmn)
    work = Vector{TF}(undef, 1)
    iwork = Vector{BlasInt}(undef, 8*minmn)
    info = Ref{BlasInt}()
    return SDDcache(U, VT, S, work, iwork, info)
end

function _checkcache(ca::SDDcache{TF}, A::AbstractMatrix{TF}) where TF
    m, n = size(A)
    minmn = min(m, n)
    size(ca.U) == (m, minmn) || throw(DimensionMismatch(
        "size of U does not match size of A"))
    size(ca.VT) == (minmn, n) || throw(DimensionMismatch(
        "size of VT does not match size of A"))
    length(ca.S) == minmn || throw(DimensionMismatch(
        "size of S does not match size of A"))
    length(ca.iwork) == 8*minmn || throw(DimensionMismatch(
        "size of iwork does not match size of A"))
end

struct SVDcache{TF<:Union{Float64, Float32}}
    U::Matrix{TF}
    VT::Matrix{TF}
    S::Vector{TF}
    work::Vector{TF}
    info::Base.RefValue{BlasInt}
end

function SVDcache(A::AbstractMatrix{TF}) where TF<:Union{Float64, Float32}
    m, n = size(A)
    minmn = min(m, n)
    U = Matrix{TF}(undef, m, minmn)
    VT = Matrix{TF}(undef, minmn, n)
    S = Vector{TF}(undef, minmn)
    work = Vector{TF}(undef, 1)
    info = Ref{BlasInt}()
    return SVDcache(U, VT, S, work, info)
end

function _checkcache(ca::SVDcache{TF}, A::AbstractMatrix{TF}) where TF
    m, n = size(A)
    minmn = min(m, n)
    size(ca.U) == (m, minmn) || throw(DimensionMismatch(
        "size of U does not match size of A"))
    size(ca.VT) == (minmn, n) || throw(DimensionMismatch(
        "size of VT does not match size of A"))
    length(ca.S) == minmn || throw(DimensionMismatch(
        "size of S does not match size of A"))
end

svdcache(::DivideAndConquer, A::AbstractMatrix) = SDDcache(A)
svdcache(::QRIteration, A::AbstractMatrix) = SVDcache(A)

for (gesdd, gesvd, elty) in
    ((:dgesdd_, :dgesvd_, :Float64),
     (:sgesdd_, :sgesvd_, :Float32))
    @eval begin
        function gesdd!(ca::SDDcache{$elty}, A::AbstractMatrix{$elty})
            Base.require_one_based_indexing(A)
            chkstride1(A)
            _checkcache(ca, A)
            m, n = size(A)
            lwork = BlasInt(-1)
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($gesdd), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                     Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                     Ptr{BlasInt}, Ptr{BlasInt}, Clong),
                     'S', m, n, A, max(1,stride(A,2)), ca.S,
                     ca.U, max(1,stride(ca.U,2)), ca.VT, max(1,stride(ca.VT,2)),
                     ca.work, lwork, ca.iwork, ca.info, 1)
                chklapackerror(ca.info[])
                if i == 1
                    # Work around issue with truncated Float32 representation of lwork in
                    # sgesdd by using nextfloat
                    lwork = round(BlasInt, nextfloat(real(ca.work[1])))
                    resize!(ca.work, lwork)
                end
            end
            return ca.U, ca.S, ca.VT
        end

        function gesvd!(ca::SVDcache{$elty}, A::AbstractMatrix{$elty})
            Base.require_one_based_indexing(A)
            chkstride1(A)
            _checkcache(ca, A)
            m, n = size(A)
            lwork = BlasInt(-1)
            for i in 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($gesvd), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                     Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                     'S', 'S', m, n, A, max(1,stride(A,2)), ca.S,
                     ca.U, max(1,stride(ca.U,2)), ca.VT, max(1,stride(ca.VT,2)),
                     ca.work, lwork, ca.info, 1, 1)
                chklapackerror(ca.info[])
                if i == 1
                    lwork = BlasInt(real(ca.work[1]))
                    resize!(ca.work, lwork)
                end
            end
            return ca.U, ca.S, ca.VT
        end
    end
end

_svd!(ca::SDDcache, A::AbstractMatrix) = gesdd!(ca, A)
_svd!(ca::SVDcache, A::AbstractMatrix) = gesvd!(ca, A)
