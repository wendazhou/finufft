include(CheckLibraryExists)


set(EXAMPLES guru1d1 guru1d1f guru2d1 many1d1 simple1d1 simple1d1f simulplans1d1 threadsafe1d1 threadsafe2d2f)
set(EXAMPLES_C guru1d1c simple1d1c simple1d1cf)

foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} PRIVATE finufft)
endforeach()

# Need to check for libm for C examples which use math.h
CHECK_LIBRARY_EXISTS(m pow "" HAVE_LIB_M)
foreach(EXAMPLE ${EXAMPLES_C})
    add_executable(${EXAMPLE} ${EXAMPLE}.c)
    target_link_libraries(${EXAMPLE} PRIVATE finufft)
    if(HAVE_LIB_M)
        target_link_libraries(${EXAMPLE} PRIVATE m)
    endif()
endforeach()
