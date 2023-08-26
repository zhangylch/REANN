subroutine get_neigh(cart,coor,atomindex,shifts,maxneigh,numatom,scutnum)
     use constant
     use initmod
     implicit none
     integer(kind=intype),intent(in) :: maxneigh,numatom
     integer(kind=intype),intent(out) :: atomindex(2,maxneigh)
     integer(kind=intype),intent(out) :: scutnum
     integer(kind=intype) :: num,iatom,ninit,i,j,l,i1,i2,i3
     integer(kind=intype) :: sca(3),boundary(2,3),rangebox(3)
     integer(kind=intype),allocatable :: index_numrs(:,:,:,:,:)
     integer(kind=intype),allocatable :: index_rs(:,:,:)
     real(kind=typenum),intent(in) :: cart(3,numatom)
     real(kind=typenum),intent(out) :: shifts(3,maxneigh)
     real(kind=typenum),intent(out) :: coor(3,numatom)
     real(kind=typenum) :: tmp
     real(kind=typenum) :: oriminv(3),orimaxv(3),tmp1(3),rangecoor(3)
     real(kind=typenum) :: fcoor(3,numatom),imageatom(3,numatom,length)
       coor=cart
       fcoor=matmul(inv_matrix,coor)
! move all atoms to an cell which is convenient for the expansion of the image
       oriminv=coor(:,1)
       orimaxv=coor(:,1)
       do iatom=2,numatom
         ! we comment out these codes to keep the original coor and these require
         ! that the coordina are in the same cell.
         sca=nint(fcoor(:,iatom)-fcoor(:,1))
         coor(:,iatom)=coor(:,iatom)-sca(1)*matrix(:,1)-sca(2)*matrix(:,2)-sca(3)*matrix(:,3)
         do j=1,3
           if(coor(j,iatom)<oriminv(j)) then
             oriminv(j)=coor(j,iatom)
           else if(coor(j,iatom)>orimaxv(j)) then
             orimaxv(j)=coor(j,iatom)
           end if
         end do
       end do
       rangecoor=orimaxv-oriminv+2.0*rc
       rangebox=ceiling(rangecoor/dier)
       oriminv=oriminv-rc
       do iatom=1,numatom
         coor(:,iatom)=coor(:,iatom)-oriminv
       end do
! obtain image 
       do l=1,length
         do iatom=1,numatom
           imageatom(:,iatom,l)=coor(:,iatom)+shiftvalue(:,l)
         end do
       end do
       allocate(index_numrs(2,numatom,rangebox(1),rangebox(2),rangebox(3)))
       allocate(index_rs(rangebox(1),rangebox(2),rangebox(3)))
       index_rs=0
       do l=1,length
         do iatom=1,numatom
           if(imageatom(1,iatom,l)>0d0 .and. imageatom(1,iatom,l)<rangecoor(1).and. &
           imageatom(2,iatom,l)>0d0 .and. imageatom(2,iatom,l)<rangecoor(2)  &
           .and. imageatom(3,iatom,l)>0d0 .and. imageatom(3,iatom,l)<rangecoor(3)) then
             sca=ceiling(imageatom(:,iatom,l)/dier)
             index_rs(sca(1),sca(2),sca(3))=index_rs(sca(1),sca(2),sca(3))+1
             index_numrs(:,index_rs(sca(1),sca(2),sca(3)),sca(1),sca(2),sca(3))=[iatom,l] 
           end if
         end do
       end do
       ninit=(length+1)/2
       do iatom = 1, numatom
         sca=ceiling(coor(:,iatom)/dier)
         ninit=(length+1)/2
         imageatom(:,iatom,ninit)=100d0
         do i=1,3
           boundary(1,i)=max(1,sca(i)-interaction)
           boundary(2,i)=min(rangebox(i),sca(i)+interaction)
         end do
         do i3=boundary(1,3),boundary(2,3)
           do i2=boundary(1,2),boundary(2,2)
             do i1=boundary(1,1),boundary(2,1)
               do i=1,index_rs(i1,i2,i3)
                 j=index_numrs(1,i,i1,i2,i3)
                 l=index_numrs(2,i,i1,i2,i3)
                 tmp1=imageatom(:,j,l)-coor(:,iatom)
                 tmp=dot_product(tmp1,tmp1)
                 if(tmp<=rcsq) then
                   scutnum=scutnum+1
                   atomindex(:,scutnum)=[iatom-1,j-1]
                   shifts(:,scutnum)=shiftvalue(:,l)
                 end if
               end do
             end do
           end do
         end do
         imageatom(:,iatom,ninit)=coor(:,iatom)
       end do
       deallocate(index_numrs)
       deallocate(index_rs)
       atomindex(1,scutnum+1:maxneigh)=0
       atomindex(2,scutnum+1:maxneigh)=1
     return
end subroutine get_neigh
   
