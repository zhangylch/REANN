    subroutine inverse_matrix(matrix,inv_matrix)
    use constant
    implicit none
    real(kind=typenum) :: matrix(3,3),inv_matrix(3,3)
    real(kind=typenum) :: tmp(3,3)
    real(kind=typenum),allocatable :: work(:)
    integer(kind=intype) :: lwork,ipiv(3),info
    lwork=-1
222 tmp=matrix
    ipiv=0
    allocate(work(max(1,lwork)))
    call dgetrf(3,3,tmp,3,ipiv,info)
    call dgetri(3,tmp,3,ipiv,work,lwork,info)
    if(lwork==-1) then
      lwork=int(work(1))
      deallocate(work)
      goto 222
    end if
    deallocate(work)
    inv_matrix=tmp
    return
    end subroutine
