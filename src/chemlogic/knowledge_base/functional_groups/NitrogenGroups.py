from neuralogic.core import R, V

from chemlogic.knowledge_base.KnowledgeBase import KnowledgeBase


class NitrogenGroups(KnowledgeBase):
    required_keys = ["layer_name", "param_size", "carbon", "oxygen", "nitrogen"]

    def create_template(self):
        # Creating a negation for carbonyl group predicate
        # self.add_rules([R.get(f"{self.layer_name}_n_carbonyl")(V.C) <= (R.get(f"{self.layer_name}_carbonyl_group")(V.C))]

        # Defining amine group (R1-C-N(-R2)-R3)
        # TODO: this won't work for datasets with implicit hydrogens (primary, secondary amines)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_amine")(V.N)
                <= (
                    # TODO: now it is relaxed because of some issue when using it (predicate has no input)
                    #  ~R.hidden.get(f"{self.layer_name}_n_carbonyl")(V.C),
                    R.get(f"{self.layer_name}_amino_group")(V.N, V.C, V.R1, V.R2)
                )
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_amino_group")(V.N, V.R1, V.R2, V.R3)
                <= (
                    R.get(self.carbon)(V.R1),
                    R.get(self.nitrogen)(V.N),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R1, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R2, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R3, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.R1, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.R2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.R3, V.B3),
                    R.special.alldiff(V.N, V.R1, V.R2, V.R3),
                )
            ]
        )

        # Defining quaternary ammonium ion (R1-N(-R2)(-R3)-R4)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_quat_ammonion")(V.N)
                <= (
                    R.get(self.nitrogen)(V.N),
                    R.get(self.carbon)(V.C),
                    R.get(f"{self.layer_name}_amino_group")(V.N, V.R1, V.R2, V.R3),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.C, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.C, V.B),
                    R.special.alldiff(V.N, V.R1, V.R2, V.R3, V.C),
                )
            ]
        )

        # Defining amide group (R-C(=O)-N(-R1)-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_amide")(V.R)
                <= (R.get(f"{self.layer_name}_amide")(V.R, V.R1, V.R2))
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_amide")(V.R1)
                <= (R.get(f"{self.layer_name}_amide")(V.R, V.R1, V.R2))
            ]
        )

        self.add_rules(
            [
                R.get(f"{self.layer_name}_amide")(V.R, V.R1, V.R2)
                <= (
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C, V.O, V.R, V.N),
                    R.get(f"{self.layer_name}_amino_group")(V.N, V.C, V.R1, V.R2),
                    R.special.alldiff(V.R, V.R1, V.R2, V.C, V.O, V.N),
                )
            ]
        )

        # Defining imine group (R1-C(=N-R)-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_imine")(V.R)
                <= R.get(f"{self.layer_name}_imine")(V.R, V.R1, V.R2)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_imine")(V.R1)
                <= R.get(f"{self.layer_name}_imine")(V.R, V.R1, V.R2)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_imine")(V.R, V.R1, V.R2)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.N, V.B),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.R1, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.R2, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.C, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.R1, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.R2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.R, V.B3),
                    R.special.alldiff(V.R, V.R1, V.R2, V.C, V.N),
                )
            ]
        )

        # Defining imide group (R1-C(=O)-N(-R)-C(=O)-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_imide")(V.R)
                <= R.get(f"{self.layer_name}_imide")(V.R, V.R1, V.R2)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_imide")(V.R1)
                <= R.get(f"{self.layer_name}_imide")(V.R, V.R1, V.R2)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_imide")(V.R, V.R1, V.R2)
                <= (
                    R.get(self.carbon)(V.C1),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.carbon)(V.C2),
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C1, V.O1, V.R1, V.N),
                    R.get(f"{self.layer_name}_carbonyl_group")(V.C2, V.O2, V.R2, V.N),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.R, V.B),
                    R.special.alldiff(V.R, V.R1, V.R2, V.C1, V.C2, V.N, V.O1, V.O2),
                )
            ]
        )

        # Defining azide group (R-N=N=N)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_azide")(V.C)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N1),
                    R.get(self.nitrogen)(V.N2),
                    R.get(self.nitrogen)(V.N3),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.N1, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N1, V.N2, V.B2),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N2, V.N3, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.N, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.N1, V.N2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.N2, V.N3, V.B3),
                    R.special.alldiff(V.C, V.N1, V.N2, V.N3),
                )
            ]
        )

        # Defining azo group (R1-N=N-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_azo")(V.C1, V.C2)
                <= (
                    R.get(self.carbon)(V.C1),
                    R.get(self.nitrogen)(V.N1),
                    R.get(self.nitrogen)(V.N2),
                    R.get(self.carbon)(V.C2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C1, V.N1, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N1, V.N2, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N2, V.C2, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.C1, V.N, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.N1, V.N2, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.N2, V.C2, V.B3),
                    R.special.alldiff(V.C1, V.C2, V.N1, V.N2),
                )
            ]
        )

        # Defining cyanate group (R-O-C≡N)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_cyanate")(V.R)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.oxygen)(V.O),
                    R.get(self.carbon)(V.R),
                    R.hidden.get(f"{self.layer_name}_triple_bonded")(V.C, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O, V.R, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.N, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.O, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.O, V.R, V.B3),
                    R.special.alldiff(V.C, V.N, V.O, V.R),
                )
            ]
        )

        # Defining isocyanate group (R-N=C=O)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_isocyanate")(V.R)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.nitrogen)(V.N),
                    R.get(self.oxygen)(V.O),
                    R.get(self.carbon)(V.R),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.C, V.O, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.R, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.N, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.O, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.R, V.B3),
                    R.special.alldiff(V.R, V.C, V.O, V.N),
                )
            ]
        )

        # Defining nitro group (R-N(=O)-O)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitro_group")(V.R)
                <= (R.get(f"{self.layer_name}_nitro_group")(V.R, V.N, V.O1, V.O2))
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitro_group")(V.R, V.N, V.O1, V.O2)
                <= (
                    R.get(self.nitrogen)(V.N),
                    R.get(self.oxygen)(V.O1),
                    R.get(self.oxygen)(V.O2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.R, V.N, V.B1),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N, V.O1, V.B2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.O2, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.R, V.N, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.O1, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.O2, V.B3),
                    R.special.alldiff(V.R, V.N, V.O1, V.O2),
                )
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitro")(V.C)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(f"{self.layer_name}_nitro_group")(V.C),
                )
            ]
        )

        # Defining nitrate group (R-O-N(=O)-O)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrate")(V.R)
                <= (R.get(f"{self.layer_name}_nitrate")(V.R, V.O, V.N, V.O1, V.O2))
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrate")(V.C, V.O, V.N, V.O1, V.O2)
                <= (
                    R.get(self.carbon)(V.C),
                    R.get(self.oxygen)(V.O),
                    R.get(f"{self.layer_name}_nitro_group")(V.O, V.N, V.O1, V.O2),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C, V.O, V.B),
                    R.get(f"{self.layer_name}_bond_message")(V.C, V.O, V.B),
                    R.special.alldiff(V.R, V.O, V.N, V.O1, V.O2, V.C),
                )
            ]
        )

        # Defining carbamate group (R-O-C(=O)-N(-R1)-R2)
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbamate")(V.R)
                <= R.get(f"{self.layer_name}_carbamate")(V.R, V.R1, V.R2)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbamate")(V.R1)
                <= R.get(f"{self.layer_name}_carbamate")(V.R, V.R1, V.R2)
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_carbamate")(V.R, V.R1, V.R2)
                <= (
                    R.get(f"{self.layer_name}_amide")(V.O, V.R1, V.R2),
                    R.get(self.oxygen)(V.O),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.O, V.R, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.O, V.R, V.B1),
                    R.special.alldiff(V.R, V.R1, V.R2, V.O),
                )
            ]
        )

        # Defining azidrine (*(C-C=N-))
        self.add_rules(
            [
                R.get(f"{self.layer_name}_aziridine")(V.C1)
                <= (
                    R.get(self.carbon)(V.C1),
                    R.get(self.carbon)(V.C1),
                    R.get(self.nitrogen)(V.N),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.C1, V.C2, V.B1),
                    R.hidden.get(f"{self.layer_name}_single_bonded")(V.N, V.C1, V.B2),
                    R.hidden.get(f"{self.layer_name}_double_bonded")(V.N, V.C2, V.B3),
                    R.get(f"{self.layer_name}_bond_message")(V.C1, V.C2, V.B1),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.C1, V.B2),
                    R.get(f"{self.layer_name}_bond_message")(V.N, V.C2, V.B3),
                    R.special.alldiff(V.C1, V.C2, V.N),
                )
            ]
        )

        # Aggregating the nitrogen groups
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_amine")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_quat_ammonion")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_amide")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_imine")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_azide")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_imide")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_azo")(V.X, V.Y)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_isocyanate")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_cyanate")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_nitro")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_nitrate")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_carbamate")(V.X)[self.param_size]
            ]
        )
        self.add_rules(
            [
                R.get(f"{self.layer_name}_nitrogen_groups")(V.X)
                <= R.get(f"{self.layer_name}_aziridine")(V.X)[self.param_size]
            ]
        )
