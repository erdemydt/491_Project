Demon : 
    Type: Interacts with two bits at a time
    Number Of States: 2, Up, Down
    Allowed Transitions:
        - All intrinsic transitions, u <-> d
        - Cooperative transitions:
            -d00 <-> u01
            -u10 <-> d11
    Features :
        - Ability to add a cooperative transition, as long as it is a viable one i.e. d_b1b2 <-> u_b1'b2'
        

Tape:
    Type: Standard Tape
    Bit States: 0, 1
    Features :
        - Ability to set the starting half of the tape to all 0s or all 1s, and the other half to the opposite value
        - Ability to set a distribution for the number "11" and "00" pairs, whatever fraction remains is filled randomly
        

When Simulated:
    - Focus is on the outcoming tape. After we check phi, we compare the inital and final tapes in a meaningful way.
    - We want to ability to compare the phi values for a normal demon and this 2-bit demon under the same conditions.
    - We want to be able to see how the correlations in the outcoming tape differ from the incoming tape for both cases.