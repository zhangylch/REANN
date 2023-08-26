module constant
     implicit none
     integer(kind=4),parameter :: intype=4,typenum=8
end module
module initmod
     use constant
     implicit none
     integer(kind=intype) :: interaction,length
     integer(kind=intype) :: nimage(3)
     real(kind=typenum) :: rc,rcsq,volume
     real(kind=typenum) :: matrix(3,3),inv_matrix(3,3)
     real(kind=typenum) :: dier ! "dier" is the side length of the box used in cell-linked 
     real(kind=typenum),allocatable :: shiftvalue(:,:)
end module

subroutine init_neigh(in_rc,in_dier,cell)
     use constant
     use initmod
     implicit none
     integer(kind=intype) :: i,j,k,l
     real(kind=typenum),intent(in) :: in_rc,in_dier,cell(3,3)
     real(kind=typenum) :: s1,s2,rlen1,rlen2
     real(kind=typenum) :: tmp(3),vec1(3,3),vec2(3,3)
       rc=in_rc
       rcsq=rc*rc
       matrix=cell
!Note that the fortran store the array with the column first, so the lattice parameters is the transpose of the its realistic shape
       tmp(1)=matrix(1,1)
       tmp(2)=matrix(2,2)
       tmp(3)=matrix(3,3)
       dier=min(in_dier,minval(tmp))+0.0001
       interaction=ceiling(rc/dier)
       nimage=ceiling(rc/abs(tmp))
       length=(2*nimage(1)+1)*(2*nimage(2)+1)*(2*nimage(3)+1)
       call inverse_matrix(matrix,inv_matrix)

! allocate the array
       allocate(shiftvalue(3,length))
! obatin image 
       vec2(:,1)=-matrix(:,1)*nimage(1)
       vec2(:,2)=-matrix(:,2)*nimage(2)
       vec2(:,3)=-matrix(:,3)*nimage(3)
       l=0
       vec1(:,3)=vec2(:,3)
       do i=-nimage(3),nimage(3)
         vec1(:,2)=vec2(:,2)
         do j=-nimage(2),nimage(2)
           vec1(:,1)=vec2(:,1)
           do k=-nimage(1),nimage(1)
             l=l+1
             shiftvalue(:,l)=vec1(:,1)+vec1(:,2)+vec1(:,3)
             vec1(:,1)=vec1(:,1)+matrix(:,1)
           end do
           vec1(:,2)=vec1(:,2)+matrix(:,2)
         end do
         vec1(:,3)=vec1(:,3)+matrix(:,3)
       end do
     return
end subroutine

subroutine deallocate_all()
     use initmod
     implicit none
       deallocate(shiftvalue)
     return
end subroutine
